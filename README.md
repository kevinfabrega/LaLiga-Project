## LaQuiniela of LaLiga

Team members: XXX - XXX -XXX

This repo contains the skeleton for you to build your first ML project. Use the data in ```laliga.sqlite``` to build a ML model that predicts the outcome of a matchday in LaLiga (Spanish Football League).

It also contains a PDF with some exercises to practice your Python skills as a Data Scientist.

### Repository structure

```
quiniela/
  ├─── logs/					# Where Logs of the program are written
  │          ...
  ├─── models/					# Where trained models are stored
  │          ...
  ├─── notebooks/				# Jupyter Notebooks used to explore the data
  │          ...
  ├─── reports/				# HTML / CSV / Excel reports
  │          ...
  ├─── src/quiniela/			# Main Python package
  |          ...
  ├─── .gitignore
  ├─── laliga.sqlite# The database
  ├─── pyproject.toml# Project's file
  └─── README.md# This file
```

### How to run it

You are provided with a fully-functional dummy model. Clone the repo and create the development environment with [uv](https://docs.astral.sh/uv/):

```console
foo@bar:~$ git clone https://github.com/tinchit0/la-quiniela.git
[... some output ...]
foo@bar:~$ cd la-quiniela
foo@bar:~/la-quiniela$ uv sync
[... some output ...]
```

This will create the virtual environment, install the dependencies and the main package. Now you can activate the environment and test the installation with the following commands:

```console
(quiniela) foo@bar:~/la-quiniela$ quiniela train --training_seasons 2010:2020
Model succesfully trained and saved in ./models/my_quiniela.model
(quiniela) foo@bar:~/la-quiniela$ quiniela predict 2021-2022 1 3
Matchday 3 - LaLiga - Division 1 - Season 2021-2022
======================================================================
         RCD Mallorca          vs            Espanyol            --> X
           Valencia            vs             Alavés             --> X
        Celta de Vigo          vs            Athletic            --> X
        Real Sociedad          vs            Levante             --> X
           Elche CF            vs           Sevilla FC           --> X
          Real Betis           vs          Real Madrid           --> X
          Barcelona            vs             Getafe             --> X
           Cádiz CF            vs           CA Osasuna           --> X
        Rayo Vallecano         vs           Granada CF           --> X
       Atlético Madrid         vs           Villarreal           --> X
```

Here, we call ```train``` to train the model using seasons from 2010 to 2020, and then we perfom a prediction of 3rd matchday of 2021-2022 season at 1st Division using ```predict```. Of course, that's a terrible prediction: that's why it's a dummy model!! Call to ```train``` did literally nothing, and ```predict``` always return ```X ```. It is your job to make something interesting.

Check out options on ```train``` and ```predict``` using ```-h``` option. You are free to add any other argument that you find necessary.

### Your job

You are asked to build a machine learning model that aims to predict the result of each match in a matchday --- either local team wins (```1```), visitor team wins (```2```) or there is a tie (```X```) ---. Use data in ```laliga.sqlite``` to build proper features and train any model you feel like the best for the case.

Use Jupyter Notebooks to play around with your model until it feels ok to you. Use all notebooks that you need, storing them in ```notebooks/``` folder. Don't panick about your code style in those notebooks, they are not going to be considered when grading. You are requested to write a final notebook called ```ModelAnalysis.ipynb```, and **this is the only one that is important** (meaning, is the only one that will be graded). This notebook must contain the training and evaluation of the final model: how good is it, where does it fail the most, what are the causes of its errors, what are the more important features and how they behave, etc. Also, export this notebook to HTML and place it in ```reports/``` folder.

Then (and only then), once you are comfortable with the result, move your code to well-structured Python modules in ```quiniela/``` folder to convert your exploratory notebooks into a functional software. You can use the provided ```QuinielaModel``` and extend it, or make it your style. Add modules to ```src/quiniela/``` folder if you need so, add parameters to ```settings.py```, add arguments to ```cli.py```. Feel like home, this skeleton is just a base for you. The project is yours.

You can use all the third-party libraries you want as long as they are available in the PyPI repositories (that is, as long as you can install them with ```pip```) and you explicitly add them in ```pyproject.toml```.

Needless to say, the model is not expected to get all the results right, far from it. Football is (luckily) highly unpredictable. If the model gets about 40%-50% right, you can already consider to be great.


### Data

The data is provided as a SQLite3 database that is inside the ZIP file. This database contains the following tables:

   * ```Matches```: All the matches played between seasons 1928-1929 and 2021-2022 with the date and score. Columns are ```season```,	```division```, ```matchday```, ```date```, ```time```, ```home_team```, ```away_team```, ```score```. Have in mind there is no time information for many of them and also that it contains matches still not played from current season.
   * ```Predictions```: The table for you to insert your predictions. It is initially empty. Columns are ```season```,	 ```timestamp```, ```division```, ```matchday```, ```home_team```, ```away_team```, ```prediction```, ```confidence```.

The data source is [Transfermarkt](https://www.transfermarkt.com/), and it was scraped using Python's library BeautifulSoup4.

