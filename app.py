from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
from plexapi.server import PlexServer
from plexapi.exceptions import NotFound
from arrapi import RadarrAPI
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, timedelta
import pytz
import os
import logging
import json
import groq
from pydantic import BaseModel
from typing import List
from translations import UI_TRANSLATIONS
from translations import TRANSLATIONS
from imdb import Cinemagoer
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import re
import requests
from abc import ABC, abstractmethod
from ollama import Client as OllamaClient
from bs4 import BeautifulSoup
import time 
import openai # Ajout pour OpenAI







from flask import redirect, url_for, render_template, request
app = Flask(__name__, static_url_path='/static')
CORS(app)

TIMEZONE = pytz.timezone(os.environ.get('TZ')) 
SETTINGS_FILE = 'user_settings.json'

scheduler = BackgroundScheduler()
scheduler.start()

collections_in_progress = {}
letterboxd_collections = {}

DEFAULT_ROOT_FOLDER = "/movies"
DEFAULT_QUALITY_PROFILE = "HD-1080p"
DEFAULT_PLEX_LIBRARY = "Films"
DEFAULT_LANG = "english"
DEFAULT_MODEL = "llama-3.1-8b-instant"
DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_LLM_API_KEY = ""
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MEDIA_SERVER = "plex"
DEFAULT_PLEX_URL = "http://localhost:32400"
DEFAULT_PLEX_TOKEN = ""
DEFAULT_JELLYFIN_URL = "http://localhost:8096"
DEFAULT_JELLYFIN_API_KEY = ""
DEFAULT_JELLYFIN_USER = ""
DEFAULT_RADARR_URL = ""
DEFAULT_RADARR_API_KEY = ""

class AIClient(ABC):
    @abstractmethod
    def get_available_models(self):
        pass

    @abstractmethod
    def is_model_available(self, model_id):
        pass

    @abstractmethod
    def chat_completion(self, messages, model, temperature=0.2):
        pass

# --- Abstraction MediaServer ---
class MediaServerClient(ABC):
    @abstractmethod
    def get_all_movies(self):
        pass
    @abstractmethod
    def movie_in_library(self, title, year=None):
        pass
    @abstractmethod
    def add_collection(self, name, movies):
        pass

class PlexClient(MediaServerClient):
    def __init__(self, plex_server):
        self.plex = plex_server
    def get_all_movies(self):
        return [m.title for m in self.plex.library.section(PLEX_LIBRARY).all()]
    def movie_in_library(self, title, year=None):
        try:
            results = self.plex.library.section(PLEX_LIBRARY).search(title)
            if year:
                return any(str(year) in str(r.year) for r in results)
            return bool(results)
        except Exception:
            return False
    def add_collection(self, name, movies):
        # Appel existant pour créer une collection Plex
        return create_plex_collection(name, movies)

class JellyfinClient(MediaServerClient):
    def __init__(self, url, api_key, user):
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.user = user
        self.headers = {"X-Emby-Token": api_key}
    def get_all_movies(self):
        r = requests.get(f"{self.url}/Users/{self.user}/Items", params={"IncludeItemTypes": "Movie"}, headers=self.headers)
        if r.status_code == 200:
            return [item['Name'] for item in r.json().get('Items', [])]
        return []
    def movie_in_library(self, title, year=None):
        movies = self.get_all_movies()
        return title in movies
    def add_collection(self, name, movies):
        # Ajout minimaliste : crée une collection vide (ajout réel à améliorer selon besoins)
        data = {"Name": name, "CollectionType": "movies"}
        r = requests.post(f"{self.url}/Collections", json=data, headers=self.headers)
        return r.status_code == 200 or r.status_code == 201

class GroqClient(AIClient):
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = groq.Client(api_key=api_key)

    def get_available_models(self):
        try:
            response = requests.get("https://api.groq.com/openai/v1/models", 
                                    headers={"Authorization": f"Bearer {self.api_key}"})
            response.raise_for_status()
            models = response.json()["data"]
            return [{"value": model["id"], "label": model["id"]} for model in models]
        except Exception as e:
            logging.error(f"Error fetching available models from Groq: {str(e)}")
            return []

    def is_model_available(self, model_id):
        try:
            response = requests.get(f"https://api.groq.com/openai/v1/models/{model_id}", 
                                    headers={"Authorization": f"Bearer {self.api_key}"})
            response.raise_for_status()
            return True
        except Exception:
            return False

    def chat_completion(self, messages, model, temperature=0.2):
        response = self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=False,
            response_format={"type": "json_object"},
        )
        # On parse le JSON dans choices[0].message.content
        import json
        content = response.choices[0].message.content
        return json.loads(content)

    


class OllamaClientWrapper(AIClient):
    def __init__(self, base_url):
        self.client = OllamaClient(host=base_url)

    def get_available_models(self):
        try:
            return [m['name'] for m in self.client.list()['models']]
        except Exception as e:
            return []

    def is_model_available(self, model_id):
        models = self.get_available_models()
        return model_id in models

    def chat_completion(self, messages, model, temperature=0.2):
        try:
            response = self.client.chat(model=model, messages=messages, options={"temperature": temperature})
            return response['message']['content']
        except Exception as e:
            return str(e)

class OpenAIClient(AIClient):
    def __init__(self, api_key):
        self.client = openai
        self.client.api_key = api_key

    def get_available_models(self):
        try:
            models = self.client.models.list()
            return [m.id for m in models.data]
        except Exception as e:
            return []

    def is_model_available(self, model_id):
        return model_id in self.get_available_models()

    def chat_completion(self, messages, model, temperature=0.2):
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

def get_ai_client_from_settings(settings):
    provider = settings.get('llm_provider', DEFAULT_LLM_PROVIDER).lower()
    if provider == 'groq':
        api_key = settings.get('groq_api_key', DEFAULT_LLM_API_KEY)
        return GroqClient(api_key)
    elif provider == 'ollama':
        ollama_url = settings.get('ollama_url', DEFAULT_OLLAMA_URL)
        return OllamaClientWrapper(ollama_url)
    elif provider == 'openai':
        api_key = settings.get('openai_api_key', DEFAULT_LLM_API_KEY)
        return OpenAIClient(api_key)
    else:
        # Par défaut, Groq (pour éviter crash si mauvais provider)
        api_key = settings.get('groq_api_key', DEFAULT_LLM_API_KEY)
        return GroqClient(api_key)


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {
        "root_folder": DEFAULT_ROOT_FOLDER,
        "quality_profile": DEFAULT_QUALITY_PROFILE,
        "plex_library": get_first_movie_library() or "Movies",
        "language": DEFAULT_LANG,
        "model": DEFAULT_MODEL,
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "llm_api_key": DEFAULT_LLM_API_KEY,
        "ollama_url": DEFAULT_OLLAMA_URL,
        "media_server": DEFAULT_MEDIA_SERVER,
        "jellyfin_url": DEFAULT_JELLYFIN_URL,
        "jellyfin_api_key": DEFAULT_JELLYFIN_API_KEY,
        "jellyfin_user": DEFAULT_JELLYFIN_USER,
        "radarr_url": DEFAULT_RADARR_URL,
        "radarr_api_key": DEFAULT_RADARR_API_KEY
    }

def write_settings_to_file(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def check_api_configurations():
    errors = []
    settings = load_settings()
    if not settings.get('plex_token') or settings.get('plex_token') == 'your_plex_token':
        errors.append('plex_token_missing')
    if not settings.get('groq_api_key') or settings.get('groq_api_key') == 'your_groq_api_key':
        errors.append('groq_api_key_missing')
    if not settings.get('radarr_api_key') or settings.get('radarr_api_key') == 'your_radarr_api_key':
        errors.append('radarr_api_key_missing')
    return errors



def settings_complete(settings):
    # Vérifie la complétude minimale pour démarrer l'app
    if not settings.get('media_server'): return False
    if settings['media_server'] == 'plex' and (not settings.get('plex_url') or not settings.get('plex_token')):
        return False
    if settings['media_server'] == 'jellyfin' and (not settings.get('jellyfin_url') or not settings.get('jellyfin_api_key') or not settings.get('jellyfin_user')):
        return False
    if not settings.get('llm_provider'): return False
    if settings['llm_provider'] == 'groq' and not settings.get('groq_api_key'):
        return False
    if settings['llm_provider'] == 'ollama' and not settings.get('ollama_url'):
        return False
    if settings['llm_provider'] == 'openai' and not settings.get('openai_api_key'):
        return False
    return True

SETTINGS = load_settings()

# Initialize Radarr client using configured URL and API key
RADARR_URL = SETTINGS.get('radarr_url', DEFAULT_RADARR_URL)
RADARR_API_KEY = SETTINGS.get('radarr_api_key', DEFAULT_RADARR_API_KEY)
radarr = RadarrAPI(RADARR_URL, RADARR_API_KEY) if RADARR_URL and RADARR_API_KEY else None

def inject_settings_into_env(settings):
    import os
    for k, v in settings.items():
        os.environ[k.upper()] = str(v)

if settings_complete(SETTINGS):
    inject_settings_into_env(SETTINGS)
    ROOT_FOLDER = SETTINGS.get('root_folder', DEFAULT_ROOT_FOLDER)
    QUALITY_PROFILE = SETTINGS.get('quality_profile', DEFAULT_QUALITY_PROFILE)
    PLEX_LIBRARY = SETTINGS.get('plex_library', DEFAULT_PLEX_LIBRARY)
    LANG = SETTINGS.get('language', DEFAULT_LANG)
    MODEL = SETTINGS.get('model', DEFAULT_MODEL)
    
    ai_client = get_ai_client_from_settings(SETTINGS)
    def get_media_server_client(settings):
        provider = (settings.get('media_server') or '').lower()
        if provider == 'jellyfin':
            url = settings.get('jellyfin_url')
            api_key = settings.get('jellyfin_api_key')
            user = settings.get('jellyfin_user')
            if not url or not api_key or not user:
                raise ValueError("Jellyfin configuration incomplete. Please complete onboarding.")
            return JellyfinClient(url=url, api_key=api_key, user=user)
        elif provider == 'plex':
            plex_url = settings.get('plex_url')
            plex_token = settings.get('plex_token')
            if not plex_url or not plex_token:
                raise ValueError("Plex configuration incomplete. Please complete onboarding.")
            if not (plex_url.startswith('http://') or plex_url.startswith('https://')):
                raise ValueError("Plex URL must start with http:// or https://. Please correct it in onboarding.")
            plex_instance = PlexServer(plex_url, plex_token)
            return PlexClient(plex_instance)
        else:
            raise ValueError("Unknown media server provider. Please complete onboarding.")
    media_server_client = get_media_server_client(SETTINGS)
    # Expose global 'plex' for functions referencing it
    if SETTINGS.get('media_server') == 'plex':
        plex = media_server_client.plex
else:
    ai_client = None
    media_server_client = None

def get_radarr_client(settings):
    from arrapi import RadarrAPI
    url = settings.get('radarr_url')
    api_key = settings.get('radarr_api_key')
    if url and api_key:
        return RadarrAPI(url, api_key)
    return None

def get_available_models():
    client = get_ai_client_from_settings(load_settings())
    if client:
        return client.get_available_models()
    return []

def is_model_available(model_id):
    client = get_ai_client_from_settings(load_settings())
    if client:
        return client.is_model_available(model_id)
    return False

def get_current_settings():
    global ROOT_FOLDER, QUALITY_PROFILE, PLEX_LIBRARY, LANG
    return {
        "root_folder": ROOT_FOLDER,
        "quality_profile": QUALITY_PROFILE,
        "plex_library": PLEX_LIBRARY,
        "language": LANG,
        "model": MODEL
    }

def is_movie_in_plex(movie_title, imdb_id):
    plex_movies = plex.library.section(SETTINGS['plex_library'])

    # Nettoyage du titre
    stripped_title = re.sub(r'\s*\(.*?\)\s*', '', movie_title).strip()
    logging.warning(f"Stripped title for search: {stripped_title}")

    # Recherche avec le titre nettoyé
    results = plex_movies.search(title=stripped_title)
    logging.warning(f"Search results: {results}")

    # Vérification des résultats de la recherche
    for movie in results:
        logging.warning(f"Plex title: {movie.title}")
        for movie_guid in movie.guids:
            logging.warning(f"GUID found: {movie_guid.id}")
            if movie_guid.id == f'imdb://tt{imdb_id}':
                logging.info(f"Found movie in Plex: {movie.title} with matching IMDb ID")
                return True

    # Si la recherche par titre n'a pas fonctionné, on parcourt tous les films
    logging.warning(f"No matching title found, searching all movies for IMDb ID {imdb_id}")
    all_movies = get_all_plex_movies()
    
    for movie in all_movies:
        results = plex_movies.search(title=movie["title"])
        for result in results:
            for movie_guid in result.guids:
                if movie_guid.id == f'imdb://tt{imdb_id}':
                    logging.info(f"Found movie in Plex: {result.title} with matching IMDb ID")
                    return True

    logging.warning(f"Movie with IMDb ID {imdb_id} not found in Plex")
    return False


def get_imdb_id(title):
    ia = Cinemagoer()

    # Recherche de tous les titres correspondants
    search_results = ia.search_movie(title)
    if search_results:
        for result in search_results:
            logging.warning(f"{title} :  {result['kind']}")
            if result['kind'] == 'movie':  # Filtrer pour ne garder que les films
                imdb_id = result.movieID
                logging.warning(f"Found IMDb ID for movie {title}: {imdb_id}")
                time.sleep(0.5)
                return imdb_id
        
        logging.warning(f"No IMDb ID found for a movie titled {title}.")
        return None
    else:
        logging.warning(f"No results found for {title}.")
        return None

def add_missing_movies_to_radarr(movies):
    added_to_radarr = []
    for movie_title in movies:
        imdb_id = get_imdb_id(movie_title)
        if imdb_id:
            radarr_movies = radarr.search_movies(imdb_id)
            if radarr_movies:
                radarr_movie = radarr_movies[0]
                if not radarr_movie.monitored:
                    radarr_movie.edit(monitored=True)
                    logging.info(f"Movie {movie_title} already in Radarr. Set to monitored.")
                added_to_radarr.append(movie_title)
                requests.post('http://localhost:9999/clear_cache')
            else:
                try:
                    new_movie = radarr.add_movie(
                        imdb_id = imdb_id,
                        root_folder = SETTINGS['root_folder'],
                        quality_profile = SETTINGS['quality_profile']
                    )
                    added_to_radarr.append(movie_title)
                    logging.info(f"Added {movie_title} to Radarr")
                except Exception as e:
                    logging.error(f"Error adding {movie_title} to Radarr: {str(e)}")
        else:
            logging.warning(f"Couldn't find IMDb ID for {movie_title}")
    return added_to_radarr


@app.route('/test_ollama')
def test_ollama():
    try:
        test_response = ai_client.chat_completion([{"role": "user", "content": "Suggest 3 science fiction movies in JSON format"}], SETTINGS['model'])
        return jsonify({"success": True, "response": test_response})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@lru_cache(maxsize=1000)
def cached_is_movie_in_plex(title, imdb_id):
    return is_movie_in_plex(title, imdb_id)

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    cached_is_movie_in_plex.cache_clear()
    return jsonify({"message": "Cache cleared"}), 200

@app.route('/search_movies', methods=['POST'])
def search_movies():
    data = request.json
    theme = data['theme']
    count = int(data['count'])
    option = data['option']
    language = SETTINGS['language']

    logging.warning(f" theme={theme}, Count={count}, Option={option}")

    try:
        recommendations = get_recommendations_from_ai(theme, count, option, language)
        if not recommendations:
            config_errors = check_api_configurations()
            if config_errors:
                return jsonify({'error': 'configuration_error', 'details': config_errors}), 400
            return jsonify({'error': 'Unable to get movie recommendations'}), 500

        def check_movie(movie):
            movie_title = movie['title']
            movie_year = str(movie['year'])
            imdb_id = get_imdb_id(f"{movie_title} ({movie_year})")
            movie['imdb_id'] = imdb_id
            
            if option in ['library', 'mixed']:
                movie['in_library'] = cached_is_movie_in_plex(movie_title, imdb_id)
            else:
                movie['in_library'] = False
            
            logging.warning(f"Film vérifié : {movie_title} ({movie_year}) - IMDb ID: {imdb_id} - Dans la bibliothèque : {movie['in_library']}")
            return movie

        with ThreadPoolExecutor(max_workers=10) as executor:
            checked_recommendations = list(executor.map(check_movie, recommendations))

        if option == 'library':
            final_recommendations = [movie for movie in checked_recommendations if movie['in_library']]
        else:  # 'mixed' ou 'discovery'
            final_recommendations = checked_recommendations

        # Limiter le nombre de résultats à 'count'
        final_recommendations = final_recommendations[:count]

        logging.warning(f"Nombre de recommandations finales : {len(final_recommendations)}")
        return jsonify({'movies': final_recommendations})
    
    except Exception as e:
        logging.error(f"Error in search_movies: {str(e)}")
        config_errors = check_api_configurations()
        if config_errors:
            return jsonify({'error': 'configuration_error', 'details': config_errors}), 400
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/create_collection', methods=['POST'])
def create_collection():
    data = request.json
    collection_name = data.get('collection_name')
    selected_movies = data.get('selected_movies', [])

    if not collection_name or not selected_movies:
        return jsonify({"error": "Missing collection name or selected movies"}), 400

    movies_in_plex = []
    movies_to_add = []
    logging.warning(selected_movies)
    for movie in selected_movies:
        imdb_id = get_imdb_id(movie)  # Vous devrez implémenter cette fonction
        if is_movie_in_plex(movie, imdb_id):
            movies_in_plex.append(movie)
        else:
            movies_to_add.append(movie)

    add_missing_movies_to_radarr(movies_to_add)
    requests.post('http://localhost:9999/clear_cache')
    collections_in_progress[collection_name] = {
        'name': collection_name,
        'movies': selected_movies,
        'added_count': len(movies_in_plex),
        'total_count': len(selected_movies),
        'status': 'En cours'
    }

    scheduler.add_job(
        check_collection_status,
        'date',
        run_date=datetime.now(TIMEZONE) + timedelta(minutes=1),
        args=[collection_name],
        id=f"check_{collection_name}",
        replace_existing=True,
        misfire_grace_time=300
    )

    return jsonify({
        "message": "Collection creation process started",
        "collection_name": collection_name,
        "movies_in_plex": movies_in_plex,
        "movies_to_add": movies_to_add
    })

@app.route('/process_letterboxd_list', methods=['POST'])
def process_letterboxd_list_route():
    data = request.json
    url = data.get('url')
    
    if not url or not url.startswith('https://letterboxd.com/'):
        return jsonify({"error": "Invalid Letterboxd URL"}), 400

    try:
        movies = get_movies_from_letterboxd(url)
        collection_name = get_letterboxd_list_title(url)
        
        movies_status = []
        for movie in movies:
            title, year = parse_movie_title(movie)
            in_plex = is_movie_in_plex_letterboxd(title, year)
            movies_status.append({
                "title": movie,
                "in_plex": in_plex
            })

        return jsonify({
            "collection_name": collection_name,
            "movies": movies_status,
            "letterboxd_url": url
        }), 200

    except Exception as e:
        print(f"Error processing Letterboxd list: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/create_letterboxd_collection', methods=['POST'])
def create_letterboxd_collection():
    data = request.json
    collection_name = data.get('collection_name')
    selected_movies = data.get('selected_movies', [])
    letterboxd_url = data.get('letterboxd_url')

    if not collection_name or not selected_movies or not letterboxd_url:
        return jsonify({"error": "Missing required data"}), 400

    try:
        # Créer la collection dans Plex
        plex_library = plex.library.section(SETTINGS['plex_library'])
        plex_movies = []
        movies_in_plex = []
        movies_to_add = []

        for movie in selected_movies:
            title, year = parse_movie_title(movie['title'])
            if movie['in_plex']:
                plex_movie = plex_library.search(title=title, year=year)[0]
                plex_movies.append(plex_movie)
                movies_in_plex.append(movie['title'])
            else:
                movies_to_add.append(movie['title'])

        if plex_movies:
            plex_collection = plex_library.createCollection(collection_name, movies=plex_movies)
            print(f"Created Plex collection: {collection_name} with {len(plex_movies)} movies")

        # Ajouter les films manquants à Radarr
        add_missing_movies_to_radarr(movies_to_add)

        # Ajouter la collection à notre application
        letterboxd_collections[collection_name] = {
            'name': collection_name,
            'url': letterboxd_url,
            'movies': [movie['title'] for movie in selected_movies],
            'last_updated': datetime.now(TIMEZONE).isoformat(),
            'is_letterboxd': True
        }

        # Planifier une mise à jour quotidienne
        scheduler.add_job(
            update_letterboxd_collection,
            'cron',
            hour=0,
            minute=1,
            args=[collection_name],
            id=f"update_letterboxd_{collection_name}",
            replace_existing=True
        )

        return jsonify({
            "message": "Letterboxd collection added successfully",
            "name": collection_name,
            "in_plex": len(movies_in_plex),
            "to_add": len(movies_to_add)
        }), 200

    except Exception as e:
        print(f"Error creating Letterboxd collection: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/delete_collection', methods=['POST'])
def delete_collection():
    data = request.json
    collection_name = data.get('name')
    
    logging.warning(f"Attempting to delete collection: {collection_name}")
    
    if not collection_name:
        logging.warning("Error: Missing collection name")
        return jsonify({"error": "Missing collection name"}), 400

    try:
        # Supprimer de Plex
        plex_library = plex.library.section(SETTINGS['plex_library'])
        logging.warning(f"Searching for collection in Plex: {collection_name}")
        collection = plex_library.collection(collection_name)
        logging.warning(f"Found collection in Plex, attempting to delete")
        collection.delete()
        logging.warning(f"Collection deleted from Plex")

        # Supprimer de notre application
        if collection_name in collections_in_progress:
            del collections_in_progress[collection_name]
            logging.warning(f"Deleted collection from collections_in_progress")
        if collection_name in letterboxd_collections:
            del letterboxd_collections[collection_name]
            logging.warning(f"Deleted collection from letterboxd_collections")

        # Supprimer la tâche planifiée si elle existe
        try:
            scheduler.remove_job(f"update_letterboxd_{collection_name}")
            logging.warning(f"Removed scheduler job for collection")
        except JobLookupError:
            logging.warning(f"No scheduler job found for collection")

        logging.warning(f"Collection {collection_name} deleted successfully")
        return jsonify({"message": "Collection deleted successfully"}), 200
    except Exception as e:
        logging.warning(f"Error deleting collection: {str(e)}")
        import traceback
        logging.warning(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
@app.route('/collections_status')
def get_collections_status():
    all_collections = list(collections_in_progress.values())
    all_collections.extend(letterboxd_collections.values())
    return jsonify(all_collections)

@app.route('/get_settings', methods=['GET'])
def get_settings():
    settings = load_settings()
    radarr = get_radarr_client(settings)
    root_folders = []
    quality_profiles = []
    radarr_error = None
    plex_libraries = []
    plex_error = None
    model_error = None

    # Radarr
    if radarr:
        try:
            root_folders = [{"value": rf.path, "label": rf.path} for rf in radarr.root_folder()]
            quality_profiles = [{"value": qp.name, "label": qp.name} for qp in radarr.quality_profile()]
        except Exception as e:
            radarr_error = f"Radarr error: {str(e)}"
            root_folders = []
            quality_profiles = []
    
    # Plex
    if settings.get('media_server') == 'plex' and settings.get('plex_url') and settings.get('plex_token'):
        try:
            from plexapi.server import PlexServer
            plex = PlexServer(settings.get('plex_url'), settings.get('plex_token'))
            plex_libraries = [{"value": section.title, "label": section.title} for section in plex.library.sections() if section.type == 'movie']
        except Exception as e:
            plex_error = f"Plex error: {str(e)}"
            plex_libraries = []
    
    # Modèles LLM
    available_models = get_available_models()
    llm_error = None
    provider = settings.get('llm_provider')
    if provider == 'groq':
        if not settings.get('groq_api_key'):
            llm_error = "Missing Groq API key."
        elif available_models == []:
            llm_error = "Failed to fetch models from Groq. Check your API key."
    elif provider == 'ollama':
        if not settings.get('ollama_url'):
            llm_error = "Missing Ollama URL."
        elif available_models == []:
            llm_error = "Failed to fetch models from Ollama. Is the server running and accessible?"
    elif provider == 'openai':
        if not settings.get('openai_api_key'):
            llm_error = "Missing OpenAI API key."
        elif available_models == []:
            llm_error = "Failed to fetch models from OpenAI. Check your API key and account status."
    
    return jsonify({
        "root_folders": root_folders,
        "quality_profiles": quality_profiles,
        "plex_libraries": plex_libraries,
        "model": available_models,
        "current_settings": settings,
        "errors": {
            "radarr": radarr_error,
            "plex": plex_error,
            "llm": llm_error
        }
    })


@app.route('/save_settings', methods=['POST'])
def save_settings():
    data = request.json
    if 'model' in data:
        if not is_model_available(data['model']):
            return jsonify({"error": "Selected model is not available"}), 400
    settings = load_settings()
    settings.update(data)
    write_settings_to_file(settings)
    return jsonify({"message": "Settings saved successfully"})

@app.route('/')
def index():
    return render_template('index.html', UI_TRANSLATIONS=UI_TRANSLATIONS)

@app.route('/manage_collections')
def manage_collections():
    return render_template('manage_collections.html', UI_TRANSLATIONS=UI_TRANSLATIONS)

app.static_folder = 'static'

# --- Onboarding route ---
@app.route('/onboarding', methods=['GET', 'POST'])
def onboarding():
    if request.method == 'POST':
        # Récupère les paramètres du formulaire et construit le dict settings
        settings = {
            'media_server': request.form.get('media_server'),
            'plex_url': request.form.get('plex_url'),
            'plex_token': request.form.get('plex_token'),
            'jellyfin_url': request.form.get('jellyfin_url'),
            'jellyfin_api_key': request.form.get('jellyfin_api_key'),
            'jellyfin_user': request.form.get('jellyfin_user'),
            'radarr_url': request.form.get('radarr_url'),
            'radarr_api_key': request.form.get('radarr_api_key'),
            'llm_provider': request.form.get('llm_provider'),
            'groq_api_key': request.form.get('groq_api_key'),
            'ollama_url': request.form.get('ollama_url'),
            'openai_api_key': request.form.get('openai_api_key'),
            # Valeurs par défaut pour la première fois
            'root_folder': '/data/FILM',
            'quality_profile': 'HD-1080p',
            'plex_library': 'Films',
            'language': 'french',
            'model': 'llama-3.1-8b-instant',
        }
        write_settings_to_file(settings)
        return redirect(url_for('index'))
    return render_template('onboarding.html')

# --- Redirection onboarding si settings incomplets ---
def settings_empty(settings):
    return all(v in ("", None) for v in settings.values())

@app.before_request
def check_onboarding():
    settings = load_settings()
    if (not settings_complete(settings) or settings_empty(settings)) \
        and request.endpoint != 'onboarding' and not request.endpoint.startswith('static'):
        return redirect(url_for('onboarding'))

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

class Movie(BaseModel):
    title: str
    year: int

class MovieList(BaseModel):
    movies: List[Movie]

def get_recommendations_from_ai(theme, count, option, language, plex_movies=None):
    translations = TRANSLATIONS.get(language, TRANSLATIONS["english"]) 
    
    if option == 'library' and plex_movies:
        prompt = translations["library_prompt"].format(movies=str(plex_movies), count=count, theme=theme)
    else:
        prompt = translations["general_prompt"].format(count=count, theme=theme)

    system_message = translations["system_message"].format(
        count=count,
        schema=json.dumps(MovieList.model_json_schema(), indent=2)
    )

    try:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ]

        # --- Choix du modèle et du provider dynamiquement ---
        # On utilise get_ai_client_from_settings(load_settings()) pour obtenir le bon client
        ai_client = get_ai_client_from_settings(load_settings())
        response = ai_client.chat_completion(messages, load_settings().get('model'), temperature=0.2)
        content = response

        logging.info(f"AI response content: {content}")
        
        if 'movies' not in content:
            raise ValueError(f"Unexpected response structure: {content}")
        
        movie_list = MovieList.model_validate(content)
        
        return [{"title": movie.title, "year": movie.year} for movie in movie_list.movies][:count]
    
    except Exception as e:
        error_message = translations["error_message"].format(error=str(e))
        logging.error(error_message)
        raise Exception("ai_error", str(e))

def get_all_plex_movies():
    try:
        plex_movies = plex.library.section(SETTINGS['plex_library'])
        return [{"title": movie.title, "year": movie.year} for movie in plex_movies.all()]
    except Exception as e:
        logging.error(f"Error fetching Plex movies: {str(e)}")
        return []
    
def movie_in_library(title, imdb_id):
    try:
        plex_movies = plex.library.section(SETTINGS['plex_library'])
        for movie in plex_movies.search(title=title):
            if any(guid.id == f'imdb://tt{imdb_id}' for guid in movie.guids):
                return True
        return False
    except Exception as e:
        logging.error(f"Error checking if movie is in library: {str(e)}")
        return False



def create_plex_collection(collection_name, movies):
    global SETTINGS
    try:
        plex_movies = plex.library.section(SETTINGS['plex_library'])
    except plexapi.exceptions.NotFound:
        logging.error(f"Error accessing Plex library '{SETTINGS['plex_library']}'. Trying to find a movie library.")
        new_library = get_first_movie_library()
        if new_library:
            SETTINGS['plex_library'] = new_library
            write_settings_to_file(SETTINGS)
            plex_movies = plex.library.section(SETTINGS['plex_library'])
        response = requests.get(url, headers=headers)
        logging.warning(f"Response status code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        movies = []
        film_posters = soup.select('li.poster-container div.film-poster')
        logging.warning(f"Number of film posters found: {len(film_posters)}")
        
        for film in film_posters:
            title = film.get('data-film-name')
            year = film.get('data-film-release-year')
            if title and year:
                movies.append(f"{title} ({year})")
            else:
                alt_text = film.find('img', class_='image')['alt']
                logging.warning(f"Using alt text for film: {alt_text}")
                movies.append(alt_text)
        
        logging.warning(f"Movies found: {len(movies)}")
        logging.warning(f"Sample movies: {movies[:5]}")
        return movies

    except Exception as e:
        print(f"Error in get_movies_from_letterboxd: {str(e)}")
        raise
    title_element = soup.select_one('h1.title-1')
    if title_element:
        return title_element.text.strip()
    else:
        logging.warning("Title element not found")
        return "Untitled List"



def is_movie_in_plex_letterboxd(movie_title, year):
    plex_movies = plex.library.section(SETTINGS['plex_library'])
    
    # Nettoyer le titre
    cleaned_title = re.sub(r'\s*\(.*?\)\s*', '', movie_title).strip()
    
    # Rechercher par titre
    results = plex_movies.search(title=cleaned_title)
    
    for movie in results:
        if movie.type == 'movie':
            # Vérifier si l'année correspond (si disponible)
            if year and str(movie.year) == str(year):
                return True
            # Si l'année n'est pas disponible, comparer juste les titres
            elif not year and movie.title.lower() == cleaned_title.lower():
                return True
    
    return False

def process_letterboxd_list(url):
    movies = get_movies_from_letterboxd(url)
    collection_name = get_letterboxd_list_title(url)
    
    movies_status = []
    
    for movie in movies:
        title, year = parse_movie_title(movie)
        in_plex = is_movie_in_plex_letterboxd(title, year)
        movies_status.append({
            "title": movie,
            "in_plex": in_plex
        })
    
    return collection_name, movies_status

def parse_movie_title(movie_string):
    match = re.match(r"(.*?)(?:\s*\((\d{4})\))?$", movie_string)
    if match:
        return match.group(1).strip(), match.group(2)
    return movie_string, None

def get_plex_movie_by_imdb(movie_title, imdb_id):
    plex_movies = plex.library.section(SETTINGS['plex_library'])

    # Nettoyage du titre
    stripped_title = re.sub(r'\s*\(.*?\)\s*', '', movie_title).strip()
    logging.warning(f"Stripped title for search: {stripped_title}")

    # Recherche avec le titre nettoyé
    results = plex_movies.search(title=stripped_title)
    logging.warning(f"Search results: {results}")

    # Vérification des résultats de la recherche
    for movie in results:
        logging.warning(f"Plex title: {movie.title}")
        for movie_guid in movie.guids:
            logging.warning(f"GUID found: {movie_guid.id}")
            if movie_guid.id == f'imdb://tt{imdb_id}':
                logging.info(f"Found movie in Plex: {movie.title} with matching IMDb ID")
                return movie.title

    # Si la recherche par titre n'a pas fonctionné, on parcourt tous les films
    logging.warning(f"No matching title found, searching all movies for IMDb ID {imdb_id}")
    all_movies = get_all_plex_movies()
    
    for movie in all_movies:
        results = plex_movies.search(title=movie["title"])
        for result in results:
            for movie_guid in result.guids:
                if movie_guid.id == f'imdb://tt{imdb_id}':
                    logging.info(f"Found movie in Plex: {result.title} with matching IMDb ID")
                    return movie.get("title")

    logging.warning(f"Movie with IMDb ID {imdb_id} not found in Plex")
    return False

def get_movies_from_letterboxd(url):
    try:    
        logging.warning(f"Fetching URL: {url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        logging.warning(f"Response status code: {response.status_code}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        movies = []
        film_posters = soup.select('li.poster-container div.film-poster')
        logging.warning(f"Number of film posters found: {len(film_posters)}")
        
        for film in film_posters:
            title = film.get('data-film-name')
            year = film.get('data-film-release-year')
            if title and year:
                movies.append(f"{title} ({year})")
            else:
                alt_text = film.find('img', class_='image')['alt']
                logging.warning(f"Using alt text for film: {alt_text}")
                movies.append(alt_text)
        
        logging.warning(f"Movies found: {len(movies)}")
        logging.warning(f"Sample movies: {movies[:5]}")
        return movies
    except Exception as e:
        print(f"Error in get_movies_from_letterboxd: {str(e)}")
        raise

def get_letterboxd_list_title(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    title_element = soup.select_one('h1.title-1')
    if title_element:
        return title_element.text.strip()
    else:
        logging.warning("Title element not found")
        return "Untitled List"

def check_collection_status(collection_name):
    global SETTINGS
    lib = plex.library.section(SETTINGS['plex_library'])
    collection = collections_in_progress.get(collection_name)
    if not collection:
        return

    added_count = 0
    all_movies_available = True

    for movie_title in collection['movies']:
        imdb_id = get_imdb_id(movie_title)
        
        if imdb_id:
            plex_movie = get_plex_movie_by_imdb(movie_title, imdb_id)
            if plex_movie:
                try:
                    search = lib.search(plex_movie)
                    search[0].addCollection(collection_name)
                    added_count += 1
                    logging.info(f"Added '{plex_movie.title}' (original title: '{movie_title}', IMDb: {imdb_id}) to collection '{collection_name}'")
                except Exception as e:
                    all_movies_available = False
                    logging.error(f"Error adding '{plex_movie.title}' (original title: '{movie_title}', IMDb: {imdb_id}) to collection: {str(e)}")
            else:
                all_movies_available = False
                logging.warning(f"Movie '{movie_title}' (IMDb: {imdb_id}) not found in the Plex library.")
        else:
            all_movies_available = False
            logging.warning(f"Couldn't find IMDb ID for '{movie_title}'")

    collection['added_count'] = added_count

    if all_movies_available:
        collection['status'] = 'Terminé'
        logging.info(f"Collection '{collection_name}' completed with {added_count} movies")
    else:
        collection['status'] = 'En cours'
        next_check = datetime.now(TIMEZONE) + timedelta(minutes=1)
        collection['next_check'] = next_check.isoformat()
        scheduler.add_job(
            check_collection_status,
            'date',
            run_date=next_check,
            args=[collection_name],
            id=f"check_{collection_name}",
            replace_existing=True,
            misfire_grace_time=300
        )

    collections_in_progress[collection_name] = collection

def update_letterboxd_collection(collection_name):
    try:
        if collection_name in letterboxd_collections:
            collection = letterboxd_collections[collection_name]
            movies = get_movies_from_letterboxd(collection['url'])
            
            plex_library = plex.library.section(SETTINGS['plex_library'])
            plex_collection = plex_library.collection(collection_name)
            
            for movie in movies:
                title, year = parse_movie_title(movie)
                if is_movie_in_plex_letterboxd(title, year):
                    plex_movie = plex_library.search(title=title, year=year)[0]
                    if plex_movie not in plex_collection.items():
                        plex_collection.addItems(plex_movie)
            
            collection['movies'] = movies
            collection['last_updated'] = datetime.now(TIMEZONE).isoformat()
            print(f"Updated Letterboxd collection: {collection_name}")
        else:
            print(f"Letterboxd collection not found: {collection_name}")
    except Exception as e:
        print(f"Error updating Letterboxd collection: {str(e)}")
        import traceback
        print(traceback.format_exc())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)