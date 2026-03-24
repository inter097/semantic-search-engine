import os
import tkinter as tk
from tkinter import ttk, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BuscadorApp:
    def __init__(self, root):
        
        self.root = root
        self.root.title("Buscador de Documentos")
        self.root.geometry("600x400")
        
        # Marco para la búsqueda
        frame_busqueda = ttk.Frame(self.root, padding="10 10 10 10")
        frame_busqueda.pack(fill=tk.X)
        
        # Etiqueta y entrada de texto
        ttk.Label(frame_busqueda, text="Buscar:").pack(side=tk.LEFT, padx=(0, 5))
        self.entrada_busqueda = ttk.Entry(frame_busqueda, width=50)
        self.entrada_busqueda.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Botón de búsqueda
        self.boton_buscar = ttk.Button(frame_busqueda, text="Buscar", command=self.buscar)
        self.boton_buscar.pack(side=tk.LEFT, padx=(5, 0))
        
        # Lista para mostrar resultados
        frame_resultados = ttk.Frame(self.root, padding="10 10 10 10")
        frame_resultados.pack(fill=tk.BOTH, expand=True)
        
        self.lista_resultados = tk.Listbox(frame_resultados, height=15)
        self.lista_resultados.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Barra de desplazamiento
        scrollbar = ttk.Scrollbar(frame_resultados, orient=tk.VERTICAL, command=self.lista_resultados.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lista_resultados.config(yscrollcommand=scrollbar.set)
        
        # Inicializar vectorizador y cargar documentos
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.documentos = {}
        self.cargar_documentos()
    
    def cargar_documentos(self):
        #ruta_documentos = os.path.join(os.getcwd(), "Minaría de textos\Tarea 3 - buscardor de archivos\text_summaries")
        ruta_documentos = r"C:\Users\eliut\OneDrive\Escritorio\Cuatrimestre_II\Minaría de textos\Tarea 3 - buscardor de archivos\text_summaries"
        
        print(f"Cargando documentos de la carpeta '{ruta_documentos}'...")

        archivos = [f for f in os.listdir(ruta_documentos) if f.endswith('.txt')]
        textos = []

        print(f"Se encontraron {len(archivos)} archivos de texto.")
        
        for archivo in archivos:
            ruta_archivo = os.path.join(ruta_documentos, archivo)
            try:
                with open(ruta_archivo, 'r', encoding='utf-8') as file:
                    contenido = file.read()
                    textos.append(contenido)
                    self.documentos[archivo] = contenido
            except Exception as e:
                print(f"Error al leer {archivo}: {e}")
        
        if textos:
            self.tf_idf_matrix = self.vectorizer.fit_transform(textos)
            self.archivos = archivos
        else:
            messagebox.showerror("Error", "No se encontraron archivos de texto en la carpeta 'documentos'.")
    
    def buscar(self):
        consulta = self.entrada_busqueda.get().strip()
        if not consulta:
            messagebox.showwarning("Entrada Vacía", "Por favor, ingresa un término de búsqueda.")
            return
        
        if not hasattr(self, 'tf_idf_matrix'):
            messagebox.showerror("Error", "No se han cargado los documentos correctamente.")
            return
        
        # Transformar la consulta en TF-IDF
        consulta_vector = self.vectorizer.transform([consulta])
        
        # Calcular la similitud del coseno
        similitudes = cosine_similarity(consulta_vector, self.tf_idf_matrix).flatten()
        
        # Obtener índices ordenados por similitud descendente
        indices_ordenados = similitudes.argsort()[::-1]
        
        # Recopilar resultados con similitud > 0
        resultados = []
        for idx in indices_ordenados:
            sim = similitudes[idx]
            if sim > 0:
                resultados.append((self.archivos[idx], sim))
            if len(resultados) >= 40:
                break
        
        # Limpiar la lista actual
        self.lista_resultados.delete(0, tk.END)
        
        if resultados:
            for archivo, similitud in resultados:
                self.lista_resultados.insert(tk.END, f"{archivo} (Similitud: {similitud:.2f})")
        else:
            self.lista_resultados.insert(tk.END, "No se encontraron resultados.")

def main():
    root = tk.Tk()
    app = BuscadorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
