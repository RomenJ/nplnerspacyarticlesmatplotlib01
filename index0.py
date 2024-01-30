import spacy
import pandas as pd
from spacy import displacy
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el modelo de inglés
nlp = spacy.load('en_core_web_sm')

# Función para cargar el contenido del archivo
def load_article(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        article_content = file.read()
    return article_content

# Ruta al archivo de texto"articles.txt"
article_file_path = "articles.txt"
#article_file_path = "uber_apple.txt"

# Cargar el artículo desde el archivo
article = load_article(article_file_path)

print("Líneas del archivo",len(article) )
# Crear un nuevo documento
doc = nlp(article)

# Visualizar en HTML
html = displacy.render(doc, style='ent', page=True)
with open('output.html', 'w', encoding='utf-8') as file:
    file.write(html)

# Crear un DataFrame para almacenar las entidades encontradas
entidadesEncontradas = pd.DataFrame(columns=['Texto', 'Categoría'])

# Llenar el DataFrame con las entidades encontradas
for ent in doc.ents:
    entidadesEncontradas = entidadesEncontradas._append({'Texto': ent.text, 'Categoría': ent.label_}, ignore_index=True)

# Contar las ocurrencias de cada categoría
conteo_categorias = entidadesEncontradas['Categoría'].value_counts()

# Graficar
plt.figure(figsize=(10, 6))
conteo_categorias.plot(kind='bar', color=plt.cm.Paired(range(len(conteo_categorias))))
plt.xlabel('Categoría')
plt.ylabel('Cantidad')
plt.title('Conteo de Categorías de Entidades')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plot.png')

# Mostrar la gráfica
plt.show()
