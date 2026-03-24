import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np

# Implementación manual de similitud del coseno
def cosine_similarity_manual(query_vector, document_vector):
    """
    Calcula la similitud del coseno entre dos vectores.
    """
    query_vector = np.array(query_vector)
    document_vector = np.array(document_vector)
    
    dot_product = np.dot(query_vector, document_vector)
    norm_query = np.linalg.norm(query_vector)
    norm_document = np.linalg.norm(document_vector)
    
    if norm_query == 0 or norm_document == 0:
        return 0.0
    
    return dot_product / (norm_query * norm_document)

class BuscadorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Buscador de Documentos")
        self.root.geometry("700x500")

        # Marco para la búsqueda
        frame_busqueda = ttk.Frame(self.root, padding="10")
        frame_busqueda.pack(fill=tk.X)

        ttk.Label(frame_busqueda, text="Buscar:").pack(side=tk.LEFT, padx=(0, 5))
        self.entrada_busqueda = ttk.Entry(frame_busqueda, width=50)
        self.entrada_busqueda.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.boton_buscar = ttk.Button(frame_busqueda, text="Buscar", command=self.buscar)
        self.boton_buscar.pack(side=tk.LEFT, padx=(5, 0))

        self.boton_cargar = ttk.Button(frame_busqueda, text="Cargar Documentos", command=self.cargar_documentos)
        self.boton_cargar.pack(side=tk.LEFT, padx=(5, 0))

        # Lista de resultados
        frame_resultados = ttk.Frame(self.root, padding="10")
        frame_resultados.pack(fill=tk.BOTH, expand=True)

        self.lista_resultados = tk.Listbox(frame_resultados, height=20)
        self.lista_resultados.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame_resultados, orient=tk.VERTICAL, command=self.lista_resultados.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lista_resultados.config(yscrollcommand=scrollbar.set)

        # Asociar evento de doble clic para abrir el archivo
        self.lista_resultados.bind("<Double-Button-1>", self.abrir_archivo)

        # Inicializar estructuras de datos
        self.documents = {}
        self.doc_vectors = []
        self.global_word_dict = {}

    def cargar_documentos(self):
        """
        Permite seleccionar una carpeta con archivos de texto y procesarlos.
        """
        folder_selected = filedialog.askdirectory()
        if not folder_selected:
            return
        
        self.documents.clear()
        self.doc_vectors.clear()
        self.global_word_dict.clear()
        
        archivos = [f for f in os.listdir(folder_selected) if f.endswith('.txt')]
        
        print(f"Se encontraron {len(archivos)} archivos de texto.")

        textos = []
        for archivo in archivos:
            ruta = os.path.join(folder_selected, archivo)
            try:
                with open(ruta, 'r', encoding='utf-8') as file:
                    contenido = file.read()
                    textos.append(contenido)
                    self.documents[archivo] = ruta
            except Exception as e:
                print(f"Error al leer {archivo}: {e}")

        # Convertir los textos en vectores numéricos
        if textos:
            self.vectorizar_textos(textos)
        else:
            messagebox.showerror("Error", "No se encontraron archivos de texto en la carpeta seleccionada.")

    def vectorizar_textos(self, textos):
        """
        Convierte los documentos en vectores numéricos.
        """
        palabras_unicas = set()
        tokenized_docs = []

        # Tokenizar documentos
        for texto in textos:
            tokens = texto.lower().split()  # Tokenización básica
            tokenized_docs.append(tokens)
            palabras_unicas.update(tokens)

        self.global_word_dict = {i: word for i, word in enumerate(sorted(palabras_unicas))}
        word_to_index = {word: i for i, word in self.global_word_dict.items()}

        # Crear matriz de frecuencias
        num_docs = len(textos)
        num_words = len(self.global_word_dict)
        self.doc_vectors = np.zeros((num_docs, num_words))

        for i, tokens in enumerate(tokenized_docs):
            for token in tokens:
                if token in word_to_index:
                    self.doc_vectors[i, word_to_index[token]] += 1  # Contar frecuencia

        print("Vectorización completada.")

    def buscar(self):
        """
        Procesa la consulta y busca los documentos más relevantes usando similitud del coseno.
        """
        consulta = self.entrada_busqueda.get().strip().lower()
        if not consulta:
            messagebox.showwarning("Entrada Vacía", "Por favor, ingresa un término de búsqueda.")
            return
        
        if not self.doc_vectors:
            messagebox.showerror("Error", "No se han cargado documentos correctamente.")
            return

        # Crear vector de consulta
        consulta_tokens = consulta.split()
        consulta_vector = np.zeros(len(self.global_word_dict))

        for token in consulta_tokens:
            for i, word in self.global_word_dict.items():
                if word == token:
                    consulta_vector[i] += 1  # Contar frecuencia

        # Calcular similitud del coseno con cada documento
        similitudes = [cosine_similarity_manual(consulta_vector, doc_vec) for doc_vec in self.doc_vectors]

        # Ordenar documentos según similitud
        archivos = list(self.documents.keys())
        ranking = sorted(zip(archivos, similitudes), key=lambda x: x[1], reverse=True)

        # Mostrar resultados en la lista
        self.lista_resultados.delete(0, tk.END)
        for archivo, sim in ranking[:40]:
            self.lista_resultados.insert(tk.END, f"{archivo} (Similitud: {sim:.4f})")

    def abrir_archivo(self, event):
        """
        Abre un archivo seleccionado en la lista de resultados.
        """
        try:
            seleccion = self.lista_resultados.curselection()
            if not seleccion:
                return
            
            archivo_seleccionado = self.lista_resultados.get(seleccion[0]).split(" (Similitud")[0]
            
            ruta_archivo = self.documents.get(archivo_seleccionado)
            if ruta_archivo and os.path.exists(ruta_archivo):
                os.startfile(ruta_archivo)  # Abre el archivo en Windows
            else:
                messagebox.showerror("Error", f"No se encontró el archivo: {archivo_seleccionado}")
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un problema al abrir el archivo: {e}")

# Inicializar la aplicación
def main():
    root = tk.Tk()
    app = BuscadorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
