# NETFLIX Movie Recommender App 🎬

Welcome to the NETFLIX Movie Recommender App, a personalized movie recommendation system that helps you discover movies based on your interests and search history.

## Features

- **User Registration and Login**: Secure registration and login using a username and 4-digit PIN.
- **Personalized Recommendations**: Get recommendations based on your recent searches.
- **Movie Information**: View detailed information about movies, including title, IMDb rating, genre, and an image.
- **Watch Trailers**: Stream movie trailers directly from the app.
- **Filter Options**: Filter movies by genres and IMDb ratings.
- **Top Picks**: Discover top picks based on your viewing history.
- **Similar Movies**: Find similar movies based on tags and summary using advanced recommendation models.

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/netflix-movie-recommender.git
    cd netflix-movie-recommender
    ```

2. **Install required packages**:

    Make sure you have Python 3.7+ installed. Then, install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the dataset**:

    Place your `movies.csv` file in the root directory of the project. This CSV file should contain movie data, including columns like Title, Actors, Tags, Summary, Genre, IMDb Score, Image, and TMDb Trailer.

4. **Run the app**:

    Start the Streamlit app with the following command:

    ```bash
    streamlit run netflix.py
    ```

## Usage

- **Register**: Create a new account by entering a username and a 4-digit PIN.
- **Login**: Log in using your username and PIN to access personalized features.
- **Search and Discover**: Use the movie selection box to search for a movie and get recommendations.
- **View Recommendations**: See top recommended movies and trailers directly in the app.

## Contributing

We welcome contributions to enhance the functionality of this app. Please feel free to fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Thanks to the [Streamlit](https://streamlit.io/) community for their support and resources.
- Movie data is provided by IMDb and TMDb.

## Contact

For any questions or feedback, please contact [Flixanalytics](mailto:flixanalytics@yahoo.com).

---

Enjoy your movie experience with the NETFLIX Movie Recommender App!
