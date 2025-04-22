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







app = Flask(__name__, static_url_path='/static')
CORS(app)
# Configuration
TIMEZONE = pytz.timezone(os.environ.get('TZ')) 
PLEX_URL = os.environ.get('PLEX_URL', 'http://localhost:32400')
PLEX_TOKEN = os.environ.get('PLEX_TOKEN', 'your_plex_token')
RADARR_URL = os.environ.get('RADARR_URL', 'http://localhost:7878')
RADARR_API_KEY = os.environ.get('RADARR_API_KEY', 'your_radarr_api_key')
MODEL_SERVER = os.environ.get('MODEL_SERVER', 'GROQ')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
OLLAMA_URL = os.environ.get('OLLAMA_URL', 'http://localhost:11434')
MODEL = os.environ.get('MODEL', 'llama-3.1-8b-instant')
plex = PlexServer(PLEX_URL, PLEX_TOKEN)
radarr = RadarrAPI(RADARR_URL, RADARR_API_KEY)
groq_client = groq.Client(api_key=GROQ_API_KEY)

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

class GroqClient(AIClient):
    def __init__(self, api_key):
        self.client = groq.Client(api_key=api_key)

    def get_available_models(self):
        try:
            response = requests.get("https://api.groq.com/openai/v1/models", 
                                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
            response.raise_for_status()
            models = response.json()["data"]
            return [{"value": model["id"], "label": model["id"]} for model in models]
        except Exception as e:
            logging.error(f"Error fetching available models from Groq: {str(e)}")
            return []

    def is_model_available(self, model_id):
        try:
            response = requests.get(f"https://api.groq.com/openai/v1/models/{model_id}", 
                                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
            response.raise_for_status()
            return True
        except Exception:
            return False

    def chat_completion(self, messages, model, temperature=0.2):
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            stream=False,
            response_format={"type": "json_object"},
        )
    


class OllamaClientWrapper(AIClient):
    def __init__(self, base_url):
        self.client = OllamaClient(host=base_url)

    def get_available_models(self):
        try:
            models = self.client.list()
            return [{"value": model["name"], "label": model["name"]} for model in models["models"]]
        except Exception as e:
            logging.error(f"Error fetching available models from Ollama: {str(e)}")
            return []

    def is_model_available(self, model_id):
        try:
            self.client.show(model_id)
            return True
        except Exception:
            return False

    def chat_completion(self, messages, model, temperature=0.2):
        try:
            # Construire le prompt
            prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            prompt += "\nReply with a JSON object containing a 'movies' array of movie objects with 'title' and 'year' properties."
            
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature
                },
                format="json",
                stream=False,
            )
            logging.info(f"Raw Ollama response: {response}")
            
            if not response or not response.get('response'):
                raise ValueError("Empty response from Ollama")
            
            content = response['response']
            logging.info(f"Extracted content: {content}")
            
            # Tentative de parsing JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse response as JSON: {content}")
                raise ValueError(f"Invalid JSON response: {content}")
        
        except Exception as e:
            logging.error(f"Error in Ollama chat completion: {str(e)}")
            raise
        
if MODEL_SERVER == 'GROQ':
    ai_client = GroqClient(GROQ_API_KEY)
elif MODEL_SERVER == 'OLLAMA':
    ai_client = OllamaClientWrapper(OLLAMA_URL)
else:
    raise ValueError(f"Unsupported MODEL_SERVER: {MODEL_SERVER}")


def get_first_movie_library():
    movie_sections = [section for section in plex.library.sections() if section.type == 'movie']
    if movie_sections:
        return movie_sections[0].title
    return None

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {
        "root_folder": DEFAULT_ROOT_FOLDER,
        "quality_profile": DEFAULT_QUALITY_PROFILE,
        "plex_library": get_first_movie_library() or "Movies",
        "language": DEFAULT_LANG,
        "model": DEFAULT_MODEL
    }

def write_settings_to_file(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

def check_api_configurations():
    errors = []
    if not PLEX_TOKEN or PLEX_TOKEN == 'your_plex_token':
        errors.append('plex_token_missing')
    if not GROQ_API_KEY or GROQ_API_KEY == 'your_groq_api_key':
        errors.append('groq_api_key_missing')
    if not RADARR_API_KEY or RADARR_API_KEY == 'your_radarr_api_key':
        errors.append('radarr_api_key_missing')
    return errors


SETTINGS = load_settings()

ROOT_FOLDER = SETTINGS['root_folder']
QUALITY_PROFILE = SETTINGS['quality_profile']
PLEX_LIBRARY = SETTINGS['plex_library']
LANG = SETTINGS['language']
MODEL = SETTINGS['model']
os.environ['LANG'] = LANG

def get_available_models():
    return ai_client.get_available_models()

def is_model_available(model_id):
    return ai_client.is_model_available(model_id)
    
def get_current_settings():
    global ROOT_FOLDER, QUALITY_PROFILE, PLEX_LIBRARY, LANG
    return {
        "root_folder": ROOT_FOLDER,
        "quality_profile": QUALITY_PROFILE,
        "plex_library": PLEX_LIBRARY,
        "language": LANG,
        "model": MODEL
    }

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

@app.route('/get_settings')
def get_settings():
    root_folders = [{"value": rf.path, "label": rf.path} for rf in radarr.root_folder()]
    quality_profiles = [{"value": qp.name, "label": qp.name} for qp in radarr.quality_profile()]
    plex_libraries = [{"value": section.title, "label": section.title} for section in plex.library.sections() if section.type == 'movie']
    available_models = get_available_models()
    
    return jsonify({
        "root_folders": root_folders,
        "quality_profiles": quality_profiles,
        "plex_libraries": plex_libraries,
        "model": available_models,
        "current_settings": SETTINGS
    })

@app.route('/save_settings', methods=['POST'])
def save_settings():
    global SETTINGS
    data = request.json
    if 'model' in data and not is_model_available(data['model']):
        return jsonify({"error": "Selected model is not available"}), 400
    SETTINGS.update(data)
    write_settings_to_file(SETTINGS)
    return jsonify({"message": "Settings saved successfully"})

@app.route('/')
def index():
    return render_template('index.html', UI_TRANSLATIONS=UI_TRANSLATIONS)

@app.route('/manage_collections')
def manage_collections():
    return render_template('manage_collections.html', UI_TRANSLATIONS=UI_TRANSLATIONS)

app.static_folder = 'static'

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
        
        if MODEL_SERVER == 'GROQ':
            response = ai_client.chat_completion(messages, SETTINGS['model'])
            content = json.loads(response.choices[0].message.content)
        else:  # OLLAMA
            response = ai_client.chat_completion(messages, SETTINGS['model'])
            content = response  # La réponse devrait déjà être un objet JSON
        
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
        else:
            logging.error("No movie library found in Plex.")
            return
    
    for movie_title in movies:
        movie = next((m for m in plex_movies.search(movie_title) if m.type == 'movie'), None)
        if movie:
            movie.addCollection(collection_name)
        else:
            logging.warning(f"Movie '{movie_title}' not found in Plex library.")

from concurrent.futures import ThreadPoolExecutor

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

def get_quality_profile_id(profile_name):
    profiles = radarr.quality_profile()
    for profile in profiles:
        if profile.name.lower() == profile_name.lower():
            return profile.id
    raise ValueError(f"Quality profile not found: {profile_name}")

def add_letterboxd_collection(url, name):
    try:
        movies = get_movies_from_letterboxd(url)
        letterboxd_collections[name] = {
            'name': name,
            'url': url,
            'movies': movies,
            'last_updated': datetime.now(TIMEZONE).isoformat(),
            'is_letterboxd': True
        }
        logging.warning(f"Added Letterboxd collection: {name} with {len(movies)} movies")
    except Exception as e:
        logging.warning(f"Error in add_letterboxd_collection: {str(e)}")
        raise

def schedule_collection_check(collection_name):
    scheduler.add_job(
        check_collection_status,
        'interval',
        minutes=1,
        args=[collection_name],
        id=f"check_{collection_name}",
        replace_existing=True
    )
    
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