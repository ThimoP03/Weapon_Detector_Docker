# CNN Prediction Docker Thimo
Deze repo behoort tot s1142306

## Voorbereiding

1. **Download en installeer Docker** volgens de instructies van Docker.
2. Alles wat tussen de <> staat, moet zelf ingevuld worden zonder de <>.

## Stappenplan

### Stap 1: Download het project
- Download het project naar je host machine.

### Stap 2: Gereed maken
- Unzip de gedownloade folder.
- Zet de te voorspellen afbeelding in een map en **onthoud de locatie**.

### Stap 3: Open een command prompt
- Open de terminal of command prompt op je machine.

### Stap 4: Navigeer naar de modelmap
- Typ het volgende commando om naar de map te gaan waar het model zich bevindt:
  
  ```bash
  cd <pad_naar_model_folder>
  ```

### Stap 5: Bouw het Docker-image
- Typ het volgende commando om het Docker-image te bouwen:

  ```bash
  docker build -t cnn_model_image  -f Docker/dockerfile .
  ```

### Stap 6: Start de Docker-container
- Start de container met de volgende opdracht, waarbij je de map met afbeeldingen koppelt:

  ```bash
  docker run -it --name <container_naam> -v <pad_naar_afbeeldingen>:/cnn-project/Prediction_Images/todo cnn_model_image bash
  ```

### Stap 7: Voer de voorspelling uit
- Binnen de Docker-bash typ je het volgende om de voorspelling te starten:

  ```bash
  python predict.py
  ```

### Stap 8: Open een nieuwe command prompt
- Open een nieuwe terminal of command prompt op je host machine.

### Stap 9: Kopieer de voorspelde afbeeldingen
- Typ het volgende commando om de voorspelde afbeeldingen van de container naar je host te kopiëren:

  ```bash
  docker cp <container_naam_of_id>:/cnn-project/Prediction_Images/done <pad_naar_folder_host>
  ```

### Stap 10: Beoordeel de voorspelde afbeeldingen
- Open de gekopieerde afbeeldingen en beoordeel de voorspellingen.


## Container opnieuw benaderen

Mocht je de container later nog moeten benaderen, maar hij staat uit, volg dan de volgende stappen:

### Stap 1: Open een command prompt
- Open een terminal of command prompt op je host machine.

### Stap 2: Start de container
- Typ het volgende commando om de container te starten:

  ```bash
  docker start <container_naam>
  ```

### Stap 3: Open de bash in de container
- Typ het volgende om toegang te krijgen tot de bash van de container:

  ```bash
  docker exec -it <container_naam> bash
  ```
