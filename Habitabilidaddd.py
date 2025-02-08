import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import threading
import time
from typing import Dict, List, Tuple, Set
import matplotlib.patches as mpatches
from tkinter import Toplevel


class CondicionesExternas:
    def __init__(self, iluminancia_exterior: float, hora_dia: int, clima: str):
        self.iluminancia_exterior = iluminancia_exterior  # en lux
        self.hora_dia = hora_dia
        self.clima = clima


class Ventana:
    def __init__(self, area: float, factor_transmision_luz: float, orientacion: str):
        self.area = area  # en m²
        self.factor_transmision_luz = factor_transmision_luz
        self.orientacion = orientacion
        self.apertura_persiana = 100  # porcentaje

    def calcular_aporte_luz_natural(self, condiciones_externas: CondicionesExternas) -> float:
        return (self.area * self.factor_transmision_luz *
                condiciones_externas.iluminancia_exterior *
                (self.apertura_persiana / 100))

    def ajustar_persianas(self, porcentaje: int):
        self.apertura_persiana = max(0, min(100, porcentaje))


class DispositivoIluminacion:
    def __init__(self, tipo: str, potencia: float, eficiencia_luminica: float,
                 temperatura_color: float):
        self.tipo = tipo
        self.potencia = potencia  # en watts
        self.eficiencia_luminica = eficiencia_luminica  # en lm/W
        self.temperatura_color = temperatura_color  # en Kelvin
        self.estado = False
        self.intensidad = 100  # porcentaje

    def encender(self):
        self.estado = True

    def apagar(self):
        self.estado = False

    def ajustar_intensidad(self, porcentaje: int):
        self.intensidad = max(0, min(100, porcentaje))

    def obtener_consumo_energetico(self) -> float:
        return self.potencia * (self.intensidad / 100) if self.estado else 0

    def obtener_flujo_luminoso(self) -> float:
        return (self.potencia * self.eficiencia_luminica *
                (self.intensidad / 100) if self.estado else 0)


class Ocupante:
    def __init__(self, id: str, tipo_actividad: str, factor_aislamiento_ropa: float):
        self.id = id
        self.tipo_actividad = tipo_actividad
        self.factor_aislamiento_ropa = factor_aislamiento_ropa
        self.preferencias_iluminacion = {}

    def establecer_preferencias(self, preferencias: Dict):
        self.preferencias_iluminacion = preferencias

    def obtener_feedback(self) -> Dict:
        return {
            'confort_visual': np.random.uniform(0, 100),
            'preferencia_temperatura_color': np.random.uniform(2700, 6500),
            'nivel_satisfaccion': np.random.uniform(0, 100)
        }

class ColoracionEspacio:
    def __init__(self, espacio: 'Espacio'):
        self.espacio = espacio
        self.grafo = {}
        self.colores = {}

    def construir_grafo_adyacencia(self):
        zonas = self._dividir_espacio_en_zonas()
        for i, zona1 in enumerate(zonas):
            self.grafo[i] = []
            for j, zona2 in enumerate(zonas[i + 1:], i + 1):
                if self._son_adyacentes(zona1, zona2):
                    self.grafo[i].append(j)

    def coloracion_greedy(self) -> Dict[int, int]:
        colores_disponibles = set(range(len(self.grafo)))  # Asegúrate de que hay suficientes colores
        nodos_ordenados = sorted(self.grafo.keys(), key=lambda x: len(self.grafo[x]), reverse=True)

        self.colores = {}
        for nodo in nodos_ordenados:
            colores_vecinos = {self.colores[vecino] for vecino in self.grafo[nodo] if vecino in self.colores}
        
        # Si no hay colores vecinos, asigna el primer color disponible
            if not colores_vecinos:
                self.colores[nodo] = next(iter(colores_disponibles))
                continue
        
        # Busca un color que no esté en los colores vecinos
            for color in colores_disponibles:
                if color not in colores_vecinos:
                    self.colores[nodo] = color
                    break

        return self.colores

    def coloracion_robusta(self, k: int) -> Dict[int, Set[int]]:
        colores_robustos = {}
        total_colores = set(range(k))
        coloracion_base = self.coloracion_greedy()

        for nodo in self.grafo.keys():
            colores_vecinos = {coloracion_base[vecino]
                               for vecino in self.grafo[nodo]}
            colores_disponibles = total_colores - colores_vecinos
            colores_robustos[nodo] = colores_disponibles
        return colores_robustos

    def _dividir_espacio_en_zonas(self, grid_size: int = 5) -> List[Dict]:
        length = width = np.sqrt(self.espacio.area)
        dx = length / grid_size
        dy = width / grid_size

        zonas = []
        for i in range(grid_size):
            for j in range(grid_size):
                zona = {
                    'centro': (i * dx + dx / 2, j * dy + dy / 2),
                    'esquinas': [
                        (i * dx, j * dy),
                        ((i + 1) * dx, j * dy),
                        (i * dx, (j + 1) * dy),
                        ((i + 1) * dx, (j + 1) * dy)
                    ]
                }
                zonas.append(zona)
        return zonas

    def _son_adyacentes(self, zona1: Dict, zona2: Dict) -> bool:
        x1, y1 = zona1['centro']
        x2, y2 = zona2['centro']
        distancia = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distancia < (np.sqrt(self.espacio.area) / 5)


class VisualizadorEspacio:
    def __init__(self, espacio: 'Espacio'):
        self.espacio = espacio
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')

    def visualizar_espacio_3d(self, espacios: List['Espacio']):
        self.ax.clear()
        for espacio in espacios:
            length = width = np.sqrt(espacio.area)
            height = 3.0

            x = np.array([[0, length], [0, length]])
            y = np.array([[0, 0], [width, width]])
            z = np.array([[0, 0], [0, 0]])
            self.ax.plot_surface(x, y, z, alpha=0.5, color='lightblue')

            self.ax.scatter(length / 2, width / 2, 0, c='black', s=100)

            zonas = espacio.coloracion._dividir_espacio_en_zonas()
            for zona in zonas:
                x_centro, y_centro = zona['centro']
                color = 'green' if espacio.es_habitable() else 'red'
                self.ax.scatter(x_centro, y_centro, 0, c=color, s=50)
                self.ax.text(x_centro, y_centro, 0.1, f"{espacio.nombre}", color='black', fontsize=10)

        self.ax.set_xlim(0, max(np.sqrt(e.area) for e in espacios))
        self.ax.set_ylim(0, max(np.sqrt(e.area) for e in espacios))
        self.ax.set_zlim(0, height)
        self.ax.set_title('Vista 3D de Todos los Espacios')

        return self.fig

    def visualizar_espacio_2d(self, espacios: List['Espacio']):
        plt.figure(figsize=(10, 8))
        for espacio in espacios:
            length = width = np.sqrt(espacio.area)
            color = 'green' if espacio.es_habitable() else 'red'
            plt.gca().add_patch(plt.Rectangle((0, 0), length, width, color=color, alpha=0.5))
            plt.text(length / 2, width / 2, espacio.nombre, horizontalalignment='center', verticalalignment='center')

        plt.xlim(0, max(np.sqrt(e.area) for e in espacios))
        plt.ylim(0, max(np.sqrt(e.area) for e in espacios))
        plt.title('Vista 2D de Todos los Espacios')
        plt.xlabel('Longitud (m)')
        plt.ylabel('Anchura (m)')
        plt.grid()
        plt.show()


class Espacio:
    def __init__(self, nombre: str, area: float, capacidad_ocupacion: int,
                 actividad_principal: str, duracion_actividad: float, 
                 x: float, y: float, z: int, es_exterior: bool = False):
        self.nombre = nombre
        self.area = area
        self.capacidad_ocupacion = capacidad_ocupacion
        self.actividad_principal = actividad_principal
        self.duracion_actividad = duracion_actividad
        self.x = x  
        self.y = y  
        self.z = z  
        self.es_exterior = es_exterior
        self.ventanas = []
        self.dispositivos_iluminacion = []
        self.ocupantes = []
        self.nivel_iluminacion = 0
        self.superficies = []
        self.coloracion = ColoracionEspacio(self)
        self.visualizador = VisualizadorEspacio(self)


    def calcular_iluminacion_total(self, condiciones_externas: CondicionesExternas) -> float:
        self.nivel_iluminacion = PropagacionLuz.calcular_iluminacion_espacio(self, condiciones_externas)
        return self.nivel_iluminacion

    def calcular_lumens(self) -> float:
        return self.nivel_iluminacion * self.area  # Lux * Area

    def es_habitable(self) -> bool:
        thresholds = {
            "Reuniones": 300,
            "Estudio": 400,
            "Descanso": 200,
        }
        if(self.actividad_principal.__eq__("Descanso")):
            return self.nivel_iluminacion <= thresholds.get(self.actividad_principal, 0)
        else:
            return self.nivel_iluminacion >= thresholds.get(self.actividad_principal, 0)


class Building:
    def __init__(self, nombre: str, floors: int, floor_width: float, floor_length: float):
        self.nombre = nombre
        self.floors = floors
        self.floor_width = floor_width
        self.floor_length = floor_length
        self.espacios = []

    def add_espacio(self, espacio: Espacio):
        if 0 <= espacio.x <= self.floor_width and 0 <= espacio.y <= self.floor_length and 0 <= espacio.z < self.floors:
            self.espacios.append(espacio)
        else:
            raise ValueError("Espacio coordinates are out of building bounds.")


class PropagacionLuz:
    @staticmethod
    def calcular_iluminacion_punto(punto: Tuple[float, float, float],
                                   fuentes_luz: List, superficies: List) -> float:
        iluminacion_directa = sum(PropagacionLuz._calcular_iluminacion_directa(punto, fuente)
                                  for fuente in fuentes_luz)
        iluminacion_indirecta = PropagacionLuz._calcular_iluminacion_indirecta(punto, superficies)
        return iluminacion_directa + iluminacion_indirecta

    @staticmethod
    def _calcular_iluminacion_directa(punto: Tuple[float, float, float],
                                      fuente) -> float:
        distancia = 1.0  # En una implementación real, calcular la distancia
        return fuente.obtener_flujo_luminoso() / (4 * math.pi * distancia ** 2)

    @staticmethod
    def _calcular_iluminacion_indirecta(punto: Tuple[float, float, float],
                                        superficies: List) -> float:
        return 50  # Valor base simplificado

    @staticmethod
    def calcular_iluminacion_espacio(espacio: Espacio, condiciones_externas: CondicionesExternas) -> float:
        iluminacion_artificial = sum(dispositivo.obtener_flujo_luminoso()
                                     for dispositivo in espacio.dispositivos_iluminacion)
        iluminacion_natural = sum(ventana.calcular_aporte_luz_natural(
            condiciones_externas)  # Valores ejemplo
            for ventana in espacio.ventanas)

        if espacio.es_exterior:
            iluminacion_natural *= 1.5  # Aumentar la luz natural en un 50% para exteriores

        return (iluminacion_artificial + iluminacion_natural) / espacio.area


class AnalizadorConfortVisual:
    @staticmethod
    def evaluar_deslumbramiento(espacio: Espacio) -> float:
        return np.random.uniform(10, 28)  # Simulación de valor UGR

    @staticmethod
    def calcular_uniformidad_iluminacion(espacio: Espacio) -> float:
        return np.random.uniform(0.4, 0.8)

    @staticmethod
    def analizar_rendimiento_color(espacio: Espacio) -> float:
        return np.random.uniform(70, 100)


class SistemaIluminacion:
    def __init__(self, building: Building):
        self.building = building  # Reference to the building
        self.dispositivos_iluminacion = []
        self.sensores_luz = []

    @property
    def espacios(self):
        return self.building.espacios  # Get spaces directly from the building

    def analizar_iluminacion_global(self, condiciones_externas: CondicionesExternas) -> Dict:
        if not self.espacios:
            print("No hay espacios en el edificio.")
            return {}

        resultados = {}
        for espacio in self.espacios:
            resultados[espacio.nombre] = {
                'nivel_iluminacion': espacio.calcular_iluminacion_total(condiciones_externas),
                'lumens': espacio.calcular_lumens(),
                'uniformidad': AnalizadorConfortVisual.calcular_uniformidad_iluminacion(espacio),
                'deslumbramiento': AnalizadorConfortVisual.evaluar_deslumbramiento(espacio),
                'rendimiento_color': AnalizadorConfortVisual.analizar_rendimiento_color(espacio),
                'habitable': espacio.es_habitable()
            }
        return resultados

    def optimizar_consumo_energetico(self) -> Dict:
        if not self.espacios:
            print("No hay espacios en el edificio para optimizar.")
            return {}

        optimizaciones = {}
        for espacio in self.espacios:
            optimizaciones[espacio.nombre] = {
                'ajuste_intensidad': self.calcular_exceso_iluminacion(espacio),
                'ahorro_estimado': self.calcular_ahorro_estimado(espacio)
            }
        return optimizaciones
    
    def get_threshold(self, actividad):
        thresholds = {
            "Reuniones": 300,
            "Estudio": 400,
            "Descanso": 200,
        }
        return thresholds.get(actividad, 0)

    def calcular_exceso_iluminacion(self, espacio : Espacio):
        threshold = self.get_threshold(espacio.actividad_principal)  
        if not espacio.dispositivos_iluminacion or all(not obj.estado for obj in espacio.dispositivos_iluminacion):
            print("The array is empty or all objects have estado=False")
            return 0

        if espacio.nivel_iluminacion <= threshold:
            return 0  
        
        R = espacio.nivel_iluminacion / threshold
        excess_percentage = (R - 1) * 100  
        return max(0, excess_percentage - 15)  
    def calcular_ahorro_estimado (self, espacio : Espacio):
        lower_ceiling = self.calcular_exceso_iluminacion(espacio) - 20
        upper_ceiling = self.calcular_exceso_iluminacion(espacio)

        return np.random.uniform(lower_ceiling, upper_ceiling)


class Recomendaciones:
    def __init__(self):
        self.recomendaciones = {
            "Reuniones": {
                "interior": {
                    "soleado": [
                        "Asegúrate de que la iluminación sea adecuada para la concentración.",
                        "Usa colores claros en las paredes para reflejar mejor la luz."
                    ],
                    "nublado": [
                        "Aumenta la luz artificial para compensar la falta de luz natural.",
                        "Usa luz blanca para mejorar la concentración."
                    ]
                },
                "exterior": {
                    "soleado": [
                        "Asegúrate de que las áreas exteriores estén bien iluminadas.",
                        "Usa sombrillas o toldos para evitar deslumbramiento."
                    ],
                    "nublado": [
                        "Aumenta la iluminación en áreas exteriores para mantener la visibilidad.",
                        "Usa luces de colores cálidos para crear un ambiente acogedor."
                    ]
                },
                "no_habitable": [
                    "Instalar iluminación adicional para alcanzar los 300 lux mínimos requeridos.",
                    "Considerar la instalación de ventanas o tragaluces para aumentar la luz natural.",
                    "Revisar y optimizar la distribución de las luminarias existentes.",
                    "Evaluar el uso de superficies reflectantes para mejorar la distribución de luz."
                ]
            },
            "Estudio": {
                "interior": {
                    "soleado": [
                        "Aumenta la luz natural abriendo las persianas.",
                        "Usa luz blanca para mejorar la concentración."
                    ],
                    "nublado": [
                        "Aumenta la luz artificial para compensar la falta de luz natural.",
                        "Usa luz blanca para mantener la concentración."
                    ]
                },
                "exterior": {
                    "soleado": [
                        "Asegúrate de que las áreas exteriores estén bien iluminadas.",
                        "Usa sombrillas o toldos para evitar deslumbramiento."
                    ],
                    "nublado": [
                        "Aumenta la iluminación en áreas exteriores para mantener la visibilidad.",
                        "Usa luces de colores cálidos para crear un ambiente acogedor."
                    ]
                },
                "no_habitable": [
                    "Instalar iluminación adicional para alcanzar los 400 lux mínimos requeridos.",
                    "Evaluar la posibilidad de añadir luz natural mediante ventanas o tragaluces.",
                    "Considerar el uso de luminarias específicas para tareas de estudio.",
                    "Revisar la disposición del mobiliario para optimizar el aprovechamiento de la luz."
                ]
            },
            "Descanso": {
                "interior": {
                    "soleado": [
                        "Usa luces suaves y cálidas para crear un ambiente relajante.",
                        "Evita la luz directa en los ojos."
                    ],
                    "nublado": [
                        "Usa luces suaves y cálidas para crear un ambiente acogedor.",
                        "Asegúrate de que la iluminación no sea demasiado brillante."
                    ]
                },
                "exterior": {
                    "soleado": [
                        "Asegúrate de que las áreas exteriores estén bien iluminadas.",
                        "Usa luces suaves para crear un ambiente relajante."
                    ],
                    "nublado": [
                        "Asegúrate de que las áreas exteriores estén bien iluminadas.",
                        "Usa luces suaves para crear un ambiente acogedor."
                    ]
                },
                "no_habitable": [
                    "Apagar iluminación adicional para bajar a menos de los 200 lux máximos requeridos.",
                    "Considerar el uso de reguladores de intensidad para ajustar la iluminación.",
                    "Evaluar la instalación de luces indirectas para crear ambientes más relajantes.",
                    "Revisar la temperatura de color de las luminarias existentes."
                ]
            }
        }

    def obtener_recomendacion(self, actividad_principal: str, es_exterior: bool, clima: str, es_habitable: bool = True) -> List[str]:
        if not es_habitable:
            return self.recomendaciones.get(actividad_principal, {}).get("no_habitable", [
                "Este espacio no cumple con los requisitos mínimos de iluminación para la actividad actual.",
                "Se recomienda:",
                "- Aumentar los niveles de iluminación",
                "- Considerar cambiar la actividad del espacio",
                "- Consultar con un especialista en iluminación"
            ])
        
        tipo_ubicacion = "exterior" if es_exterior else "interior"
        return self.recomendaciones.get(actividad_principal, {}).get(tipo_ubicacion, {}).get(clima, [
            "No hay recomendaciones específicas disponibles para esta combinación de actividad, ubicación y clima.",
            "Por favor, consulte las guías generales de iluminación."
        ])
        
class AdjacencyChecker:
    def __init__(self, spaces, threshold=2):
        self.spaces = spaces
        self.threshold = threshold
        self.adjacency_matrix_3d = self.build_adjacency_matrix_3d()
        self.adjacency_matrix_2d = self.build_adjacency_matrix_2d()

    def is_adjacent_3d(self, espacio1, espacio2):
        distance = np.linalg.norm([espacio1.x - espacio2.x, espacio1.y - espacio2.y, espacio1.z - espacio2.z])
        return distance <= self.threshold

    def is_adjacent_2d(self, espacio1, espacio2):
        if espacio1.z != espacio2.z:
            return False
        distance = np.linalg.norm([espacio1.x - espacio2.x, espacio1.y - espacio2.y])
        return distance <= self.threshold

    def build_adjacency_matrix_3d(self):
        n = len(self.spaces)
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if self.is_adjacent_3d(self.spaces[i], self.spaces[j]):
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        return matrix

    def build_adjacency_matrix_2d(self):
        n = len(self.spaces)
        matrix = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if self.is_adjacent_2d(self.spaces[i], self.spaces[j]):
                    matrix[i][j] = 1
                    matrix[j][i] = 1
        return matrix
    
    def update_adjacency_matrices(self):
        n = len(self.spaces)
        self.adjacency_matrix_3d = np.zeros((n, n), dtype=int)
        self.adjacency_matrix_2d = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                if self.is_adjacent_3d(self.spaces[i], self.spaces[j]):
                    self.adjacency_matrix_3d[i][j] = 1
                    self.adjacency_matrix_3d[j][i] = 1
                
                if self.is_adjacent_2d(self.spaces[i], self.spaces[j]):
                    self.adjacency_matrix_2d[i][j] = 1
                    self.adjacency_matrix_2d[j][i] = 1

class InterfazGrafica:
    def __init__(self, sistema: 'SistemaIluminacion'):
        self.sistema = None
        self.recomendaciones = Recomendaciones()
        self.adjacency_checker = None
        self.root = tk.Tk()
        self.estilizar_interfaz()
        self.root.title("Sistema de Habitabilidad")
        self.root.geometry("1920x1080")
        self.root.configure(bg="#f0f0f0")
        self.simulador = None

        # Crear un sistema de pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Pestañas
        self.simulacion_tab = ttk.Frame(self.notebook)
        self.configuracion_tab = ttk.Frame(self.notebook)
        self.analisis_tab = ttk.Frame(self.notebook)
        self.optimizacion_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.simulacion_tab, text="Simulación")
        self.notebook.add(self.configuracion_tab, text="Configuración")
        self.notebook.add(self.analisis_tab, text="Análisis")
        self.notebook.add(self.optimizacion_tab, text="Optimización")

        # Configurar pestañas
        self.configurar_simulacion_tab()
        self.configurar_configuracion_tab()
        self.configurar_analisis_tab()
        self.configurar_optimizacion_tab()

        self.root.mainloop()

    def estilizar_interfaz(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=6, relief="flat", background="#4CAF50", foreground="white", font=("Helvetica", 12))
        style.map("TButton", background=[("active", "#45a049")])
        style.configure("TLabel", background="#f0f0f0", font=("Helvetica", 12))
        style.configure("TNotebook", background="#f0f0f0", font=("Helvetica", 12))
        style.configure("TNotebook.Tab", padding=[10, 5], font=("Helvetica", 12))

    def configurar_simulacion_tab(self):
        self.simulacion_frame = ttk.Frame(self.simulacion_tab, padding="10")
        self.simulacion_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.simulacion_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.control_frame = ttk.Frame(self.simulacion_frame)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.panel_simulacion = ttk.Frame(self.control_frame)
        self.panel_simulacion.pack(pady=10)

        ttk.Label(self.panel_simulacion, text="Hora del día:").grid(row=0, column=0, pady=5)

        # Label to display simulated time
        self.time_label = ttk.Label(self.panel_simulacion, text="Hora no disponible", font=("Arial", 12))
        self.time_label.grid(row=0, column=1, pady=5)

        # Start updating the time in GUI
        self.actualizar_hora_gui()

        ttk.Label(self.panel_simulacion, text="Clima:").grid(row=1, column=0, pady=5)
        self.clima_var = tk.StringVar(value="despejado")
        ttk.Combobox(self.panel_simulacion, values=["despejado", "nublado"], textvariable=self.clima_var).grid(row=1, column=1, pady=5)

        self.crear_botones()

    def actualizar_hora_gui(self):
        """ Updates the time label in the simulation panel, only if the simulator is initialized. """
        if hasattr(self, 'simulador') and self.simulador is not None:
            self.time_label.config(text=f"{self.simulador.obtener_hora_actual()}")
        else:
            self.time_label.config(text="Hora no disponible")

        self.root.after(1000, self.actualizar_hora_gui)


    def mostrar_informacion_algoritmos(self):
        ventana_info = tk.Toplevel(self.root)
        ventana_info.title("Información de Algoritmos de Coloración")

        texto = tk.Text(ventana_info, wrap=tk.WORD, width=60, height=20)
        texto.pack(padx=10, pady=10)

        texto.insert(tk.END, "Coloración Greedy:\n")
        texto.insert(tk.END, "Este algoritmo asigna colores a las zonas de manera que no haya dos zonas adyacentes con el mismo color. Es rápido y eficiente, pero no siempre encuentra la solución óptima.\n\n")
        texto.insert(tk.END, "Coloración Robusta:\n")
        texto.insert(tk.END, "Este algoritmo extiende la coloración greedy para asegurar que cada zona tenga un conjunto de colores disponibles, aumentando la robustez del sistema ante cambios o fallos. Es más complejo y puede ser más lento, pero ofrece una mayor flexibilidad.\n\n")

        ttk.Button(ventana_info, text="Cerrar", command=ventana_info.destroy).pack(pady=10)


    def configurar_configuracion_tab(self):
        self.configuracion_frame = ttk.Frame(self.configuracion_tab, padding="10")
        self.configuracion_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(self.configuracion_frame, text="Agregar Espacio", command=self.mostrar_dialogo_espacio).pack(pady=5)
        ttk.Button(self.configuracion_frame, text="Agregar Dispositivo", command=self.mostrar_dialogo_dispositivo).pack(pady=5)
        ttk.Button(self.configuracion_frame, text="Agregar Edificio", command=self.mostrar_guardar_edificio).pack(pady=5)
        ttk.Button(self.configuracion_frame, text="Modificar Espacio", command=self.mostrar_dialogo_modificar_espacio).pack(pady=5)

    def configurar_analisis_tab(self):
        self.analisis_frame = ttk.Frame(self.analisis_tab, padding="10")
        self.analisis_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(self.analisis_frame, text="Analizar Iluminación", command=self.mostrar_analisis).pack(pady=5)

    def configurar_optimizacion_tab(self):
        self.optimizacion_frame = ttk.Frame(self.optimizacion_tab, padding="10")
        self.optimizacion_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(self.optimizacion_frame, text="Optimizar Consumo", command=self.mostrar_optimizacion).pack(pady=5)
        
    def crear_botones(self):
        # Botones de control en la pestaña de simulación
        ttk.Button(self.control_frame, text="Coloración Greedy", command=self.plot_3d_popup_no_restriction).pack(pady=5)
        ttk.Button(self.control_frame, text="Coloración Robusta", command=self.plot_3d_popup_robust).pack(pady=5)
        ttk.Button(self.control_frame, text="Vista 3D", command=self.plot_3d_popup).pack(pady=5)
        ttk.Button(self.control_frame, text="Vista 2D", command=self.plot_2d_popup).pack(pady=5)
        ttk.Button(self.control_frame, text="Control de Dispositivos", command=self.mostrar_control_dispositivos).pack(pady=5)

    def greedy_coloring(self):
        n = len(self.sistema.building.espacios)
        colors = [-1] * n  
        
        for i in range(n):
            adjacent_colors = set()
            for j in range(n):
                if self.adjacency_checker.adjacency_matrix_2d[i][j] == 1 and colors[j] != -1:
                    adjacent_colors.add(colors[j])

        
            color = 0
            while color in adjacent_colors:
                color += 1
            self.sistema.building.espacios[i].calcular_iluminacion_total(CondicionesExternas(self.simulador.iluminancia_exterior, self.simulador.hora_dia, self.simulador.clima))
            if self.sistema.building.espacios[i].es_habitable():
                
                colors[i] = 'g'  
            else:
                offset_pct = (self.sistema.building.espacios[i].nivel_iluminacion / self.get_threshold(self.sistema.building.espacios[i].actividad_principal)) * 100
                if abs(offset_pct - 100) < 10:
                    colors[i] = 'y'  
                else:
                    
                    colors[i] = 'r' 

        return colors

    def get_threshold(self, actividad):
        thresholds = {
            "Reuniones": 300,
            "Estudio": 400,
            "Descanso": 200,
        }
        return thresholds.get(actividad, 0)

    
    def plot_2d_popup(self):
        """Create a pop-up window for the 2D plot"""
        popup = Toplevel(self.root)
        popup.title("Vista 2D")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Vista 2D de los espacios y conexiones (Por piso)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_xlim(0, self.sistema.building.floor_width)
        ax.set_ylim(0, self.sistema.building.floor_length)

        colors = self.greedy_coloring()

        for i, espacio in enumerate(self.sistema.building.espacios):
            ax.scatter(espacio.x, espacio.y + 0.2, color=colors[i], label=espacio.nombre if i == 0 else "")
            ax.text(espacio.x, espacio.y, espacio.nombre, horizontalalignment='center', verticalalignment='center')
            for j, is_adj in enumerate(self.adjacency_checker.adjacency_matrix_2d[i]):
                if is_adj:
                    ax.plot([espacio.x, self.sistema.building.espacios[j].x],
                            [espacio.y, self.sistema.building.espacios[j].y], 'r-')

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack()
        canvas.draw()


    def greedy_coloring_3d(self):
        n = len(self.sistema.building.espacios)
        colors = [-1] * n  
        
        for i in range(n):
            adjacent_colors = set()
            for j in range(n):
                if self.adjacency_checker.adjacency_matrix_3d[i][j] == 1 and colors[j] != -1:
                    adjacent_colors.add(colors[j])

            color = 0
            while color in adjacent_colors:
                color += 1

            self.sistema.building.espacios[i].calcular_iluminacion_total(CondicionesExternas(self.simulador.iluminancia_exterior, self.simulador.hora_dia, self.simulador.clima))
            
            if self.sistema.building.espacios[i].es_habitable():
                colors[i] = 'g'  # Green if habitable
            else:
                offset_pct = (self.sistema.building.espacios[i].nivel_iluminacion / self.get_threshold(self.sistema.building.espacios[i].actividad_principal)) * 100
                if abs(offset_pct - 100) < 10:
                    colors[i] = 'y'  # Yellow if within 10% of the threshold
                else:
                    colors[i] = 'r'  # Red if non-habitable

        return colors
    

    def plot_3d_popup(self):
        popup = Toplevel(self.root)
        popup.title("Vista 3D")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Vista 3D de todos los espacios y conexiones")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(0, self.sistema.building.floor_width)
        ax.set_ylim(0, self.sistema.building.floor_length)
        ax.set_zlim(0, self.sistema.building.floors)

        colors = self.greedy_coloring_3d()

        for i, espacio in enumerate(self.sistema.building.espacios):
            ax.scatter(espacio.x, espacio.y, espacio.z, color=colors[i], label=espacio.nombre if i == 0 else "")
            ax.text(espacio.x, espacio.y, espacio.z + 0.2, espacio.nombre, color='black', fontsize=8)

            for j, is_adj in enumerate(self.adjacency_checker.adjacency_matrix_3d[i]):
                if is_adj:
                    ax.plot([espacio.x, self.sistema.building.espacios[j].x],
                            [espacio.y, self.sistema.building.espacios[j].y],
                            [espacio.z, self.sistema.building.espacios[j].z], 'r-')

        # Legend for color classification
        legend_patches = [
            mpatches.Patch(color='g', label='Habitable'),
            mpatches.Patch(color='y', label='Casi habitable'),
            mpatches.Patch(color='r', label='No habitable')
        ]
        ax.legend(handles=legend_patches, loc='upper right')

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def greedy_coloring_3d_no_restriction(self):
        """Applies a greedy coloring algorithm to the 3D adjacency graph."""
        n = len(self.sistema.building.espacios)
        colors = [-1] * n  
        
        for i in range(n):
            adjacent_colors = set()
            
            for j in range(n):
                if self.adjacency_checker.adjacency_matrix_3d[i][j] == 1 and colors[j] != -1:
                    adjacent_colors.add(colors[j])

            color = 0
            while color in adjacent_colors:
                color += 1
            
            colors[i] = color 

        return colors
    

    def plot_3d_popup_no_restriction(self):
        """Displays a 3D visualization of the building with greedy-colored nodes."""
        popup = Toplevel(self.root)
        popup.title("Vista 3D")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Vista 3D de la coloracion del grafo")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(0, self.sistema.building.floor_width)
        ax.set_ylim(0, self.sistema.building.floor_length)
        ax.set_zlim(0, self.sistema.building.floors)

        colors = self.greedy_coloring_3d_no_restriction()

        # Define a colormap for better visualization
        cmap = plt.get_cmap("tab10")  # A colormap with 10 distinguishable colors

        for i, espacio in enumerate(self.sistema.building.espacios):
            color_index = colors[i] % 10  # Ensure we stay within the colormap range
            color = cmap(color_index)

            ax.scatter(espacio.x, espacio.y, espacio.z, color=color, label=espacio.nombre if i == 0 else "")
            ax.text(espacio.x, espacio.y, espacio.z + 0.2, espacio.nombre, color='black', fontsize=8)

            # Draw adjacency connections
            for j, is_adj in enumerate(self.adjacency_checker.adjacency_matrix_3d[i]):
                if is_adj:
                    ax.plot([espacio.x, self.sistema.building.espacios[j].x],
                            [espacio.y, self.sistema.building.espacios[j].y],
                            [espacio.z, self.sistema.building.espacios[j].z], 'k-', alpha=0.5)  # Light gray edges

        # Create a legend based on the assigned colors

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack()
        canvas.draw()
    
    def robust_greedy_coloring_3d(self):
    
        n = len(self.sistema.building.espacios)
        colors = [-1] * n  # Initialize all nodes with no color

        # Sort nodes by degree (number of connections) in descending order
        node_degrees = [(i, sum(self.adjacency_checker.adjacency_matrix_3d[i])) for i in range(n)]
        sorted_nodes = sorted(node_degrees, key=lambda x: x[1], reverse=True)  # Sort by degree

        for node, _ in sorted_nodes:
            adjacent_colors = set()

            # Collect colors of already colored adjacent nodes
            for neighbor in range(n):
                if self.adjacency_checker.adjacency_matrix_3d[node][neighbor] == 1 and colors[neighbor] != -1:
                    adjacent_colors.add(colors[neighbor])

            # Assign the lowest possible color that is not in adjacent_colors
            color = 0
            while color in adjacent_colors:
                color += 1

            colors[node] = color  # Assign the chosen color

        return colors

    def plot_3d_popup_robust(self):
        """Displays a 3D visualization of the building with greedy-colored nodes."""
        popup = Toplevel(self.root)
        popup.title("Vista 3D")

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Vista 3D de la coloracion del grafo")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(0, self.sistema.building.floor_width)
        ax.set_ylim(0, self.sistema.building.floor_length)
        ax.set_zlim(0, self.sistema.building.floors)

        colors = self.robust_greedy_coloring_3d()

        # Define a colormap for better visualization
        cmap = plt.get_cmap("tab10")  # A colormap with 10 distinguishable colors

        for i, espacio in enumerate(self.sistema.building.espacios):
            color_index = colors[i] % 10  # Ensure we stay within the colormap range
            color = cmap(color_index)

            ax.scatter(espacio.x, espacio.y, espacio.z, color=color, label=espacio.nombre if i == 0 else "")
            ax.text(espacio.x, espacio.y, espacio.z + 0.2, espacio.nombre, color='black', fontsize=8)

            # Draw adjacency connections
            for j, is_adj in enumerate(self.adjacency_checker.adjacency_matrix_3d[i]):
                if is_adj:
                    ax.plot([espacio.x, self.sistema.building.espacios[j].x],
                            [espacio.y, self.sistema.building.espacios[j].y],
                            [espacio.z, self.sistema.building.espacios[j].z], 'k-', alpha=0.5)  # Light gray edges

        # Create a legend based on the assigned colors

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.get_tk_widget().pack()
        canvas.draw()

    def mostrar_guardar_edificio(self):
        """Creates a pop-up window for adding a building."""
        dialogo = tk.Toplevel(self.root)
        dialogo.title("Agregar Edificio")

        campos = {
            "Nombre": "",
            "Largo": "20.0",
            "Ancho": "20.0",
            "Pisos": "3",
            "Espacios por piso" : "2"
        }

        entries = {}
        row = 0
        for campo, valor in campos.items():
            ttk.Label(dialogo, text=campo).grid(row=row, column=0, pady=5)
            entry = ttk.Entry(dialogo)
            entry.insert(0, valor)
            entry.grid(row=row, column=1, pady=5)
            entries[campo] = entry
            row += 1

        # Button to save the building
        btn_guardar = ttk.Button(dialogo, text="Guardar", 
                                 command=lambda: self.guardar_edificio(
                                     entries["Nombre"].get(), 
                                     int(entries["Pisos"].get()), 
                                     float(entries["Ancho"].get()), 
                                     float(entries["Largo"].get()),
                                     int(entries["Espacios por piso"].get()),
                                     dialogo))
        btn_guardar.grid(row=row, columnspan=2, pady=10)

    def guardar_edificio(self, name: str, floors: int, floor_width: float, floor_length: float, room_number: int, dialog):
        self.building = Building(name, floors, floor_width, floor_length)

        actividades = ["Reuniones", "Estudio", "Descanso"]  # Default activities
        espacio_id = 1  # Unique ID for naming
        proximity_threshold = 2.0  # Define how close to the edges an Espacio should be to get a window

        for z in range(floors):  # Loop over floors
            for _ in range(room_number):  # Create two spaces per floor
                x = np.random.uniform(0, floor_width)
                y = np.random.uniform(0, floor_length)

                espacio = Espacio(
                    nombre=f"Espacio {espacio_id}",
                    area=25,
                    capacidad_ocupacion=np.random.choice([5, 8]),  # Randomly assign capacity
                    actividad_principal=actividades[espacio_id % len(actividades)],
                    duracion_actividad=np.random.choice([2, 4]),  # Randomly assign duration
                    x=x,
                    y=y,
                    z=z
                )

                # Check if the space is near any edges
                if x <= proximity_threshold:
                    espacio.ventanas.append(Ventana(area=1.5, factor_transmision_luz=0.7, orientacion="Oeste"))
                elif x >= (floor_width - proximity_threshold):
                    espacio.ventanas.append(Ventana(area=1.5, factor_transmision_luz=0.7, orientacion="Este"))

                if y <= proximity_threshold:
                    espacio.ventanas.append(Ventana(area=1.5, factor_transmision_luz=0.7, orientacion="Norte"))
                elif y >= (floor_length - proximity_threshold):
                    espacio.ventanas.append(Ventana(area=1.5, factor_transmision_luz=0.7, orientacion="Sur"))

                self.building.add_espacio(espacio)
                espacio_id += 1

        # Now that the building has spaces, initialize the system and adjacency checker
        self.sistema = SistemaIluminacion(self.building)
        self.simulador = SimuladorCondicionesExternas(self.sistema)
        self.simulador.iniciar_simulacion()

        self.adjacency_checker = AdjacencyChecker(self.building.espacios)

        messagebox.showinfo("Éxito", f"Edificio '{name}' creado con {floors * 2} espacios.")

        dialog.destroy()


    def mostrar_dialogo_modificar_espacio(self):
        """Opens a dialog to modify an existing Espacio."""
        if not self.sistema.building.espacios:
            messagebox.showerror("Error", "No hay espacios para modificar.")
            return

        dialogo = tk.Toplevel(self.root)
        dialogo.title("Modificar Espacio")

        # Dropdown to select Espacio
        ttk.Label(dialogo, text="Seleccionar Espacio:").grid(row=0, column=0, pady=5)
        espacio_var = tk.StringVar()
        espacio_combo = ttk.Combobox(dialogo, textvariable=espacio_var, state="readonly")
        espacio_combo["values"] = [espacio.nombre for espacio in self.sistema.building.espacios]
        espacio_combo.grid(row=0, column=1, pady=5)

        # Fields for modification (excluding area and exterior)
        campos = {
            "Nombre": "",
            "Capacidad": "0",
            "Duración Actividad (h)": "0.0"
        }

        entries = {}
        row = 1
        for campo, valor in campos.items():
            ttk.Label(dialogo, text=campo).grid(row=row, column=0, pady=5)
            entry = ttk.Entry(dialogo)
            entry.grid(row=row, column=1, pady=5)
            entries[campo] = entry
            row += 1

        # Activity Dropdown
        ttk.Label(dialogo, text="Actividad Principal:").grid(row=row, column=0, pady=5)
        actividad_var = tk.StringVar()
        actividad_combo = ttk.Combobox(dialogo, textvariable=actividad_var)
        actividad_combo["values"] = ["Reuniones", "Estudio", "Descanso"]
        actividad_combo.grid(row=row, column=1, pady=5)

        def cargar_datos_espacio(event):
            """Loads the selected Espacio's data into the input fields."""
            selected_nombre = espacio_var.get()
            espacio = next((e for e in self.sistema.building.espacios if e.nombre == selected_nombre), None)
            
            if espacio:
                entries["Nombre"].delete(0, tk.END)
                entries["Nombre"].insert(0, espacio.nombre)
                
                entries["Capacidad"].delete(0, tk.END)
                entries["Capacidad"].insert(0, str(espacio.capacidad_ocupacion))
                
                actividad_var.set(espacio.actividad_principal)

                entries["Duración Actividad (h)"].delete(0, tk.END)
                entries["Duración Actividad (h)"].insert(0, str(espacio.duracion_actividad))

        espacio_combo.bind("<<ComboboxSelected>>", cargar_datos_espacio)

        def guardar_cambios():
            """Saves the modified data back to the selected Espacio object."""
            try:
                selected_nombre = espacio_var.get()
                espacio = next((e for e in self.sistema.building.espacios if e.nombre == selected_nombre), None)
                
                if not espacio:
                    raise ValueError("Espacio no encontrado.")

                nuevo_nombre = entries["Nombre"].get().strip()
                if not nuevo_nombre:
                    raise ValueError("El nombre no puede estar vacío.")

                espacio.nombre = nuevo_nombre
                espacio.capacidad_ocupacion = int(entries["Capacidad"].get())
                espacio.actividad_principal = actividad_var.get()
                espacio.duracion_actividad = float(entries["Duración Actividad (h)"].get())

                # Update adjacency and visualization
                self.adjacency_checker.update_adjacency_matrices()
                self.actualizar_visualizacion()
                
                dialogo.destroy()
            except ValueError as e:
                messagebox.showerror("Error", str(e))

        # Save button
        ttk.Button(dialogo, text="Guardar Cambios", command=guardar_cambios).grid(row=row + 1, column=0, columnspan=2, pady=10)

    def mostrar_control_dispositivos(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para controlar dispositivos")
            return

        dialogo = tk.Toplevel(self.root)
        dialogo.title("Control de Dispositivos de Iluminación")

        dispositivo_var = tk.StringVar()
        dispositivo_combo = ttk.Combobox(dialogo, textvariable=dispositivo_var)
        dispositivo_combo['values'] = [f"{dispositivo.tipo} (Estado: {'Encendido' if dispositivo.estado else 'Apagado'})"
                                        for espacio in self.sistema.espacios
                                        for dispositivo in espacio.dispositivos_iluminacion]
        dispositivo_combo.grid(row=0, column=0, pady=5)

        ttk.Button(dialogo, text="Encender", command=lambda: self.control_dispositivo(dispositivo_var.get(), True)).grid(row=1, column=0, pady=5)
        ttk.Button(dialogo, text="Apagar", command=lambda: self.control_dispositivo(dispositivo_var.get(), False)).grid(row=2, column=0, pady=5)

    def control_dispositivo(self, dispositivo_info: str, encender: bool):
        tipo_dispositivo = dispositivo_info.split(" (")[0]
        for espacio in self.sistema.espacios:
            for dispositivo in espacio.dispositivos_iluminacion:
                if dispositivo.tipo == tipo_dispositivo:
                    if encender:
                        dispositivo.encender()
                    else:
                        dispositivo.apagar()
                    messagebox.showinfo("Control de Dispositivo", f"{tipo_dispositivo} ha sido {'encendido' if encender else 'apagado'}.")
                    return
        messagebox.showwarning("Advertencia", "Dispositivo no encontrado.")

    def mostrar_dialogo_espacio(self):
        dialogo = tk.Toplevel(self.root)
        dialogo.title("Agregar Nuevo Espacio")

        campos = {
            "Nombre": "",
            "Área (m²)": "0.0",
            "Capacidad": "0",
            "Actividad Principal": "",
            "Duración Actividad (h)": "0.0",
            "Es Exterior": "No"
        }

        row = 0
        entries = {}
        for campo, valor in campos.items():
            ttk.Label(dialogo, text=campo).grid(row=row, column=0, pady=5)
            entry = ttk.Entry(dialogo)
            entry.insert(0, valor)
            entry.grid(row=row, column=1, pady=5)
            entries[campo] = entry
            row += 1

        ttk.Label(dialogo, text="Actividad Principal:").grid(row=row, column=0, pady=5)
        actividad_var = tk.StringVar()
        actividad_combo = ttk.Combobox(dialogo, textvariable=actividad_var)
        actividad_combo['values'] = ["Reuniones", "Estudio", "Descanso"]
        actividad_combo.grid(row=row, column=1, pady=5)

        def guardar_espacio():
            try:
                nombre = entries["Nombre"].get().strip()
                if not nombre:
                    raise ValueError("El nombre del espacio no puede estar vacío.")

                area = float(entries["Área (m²)"].get())
                if area <= 0:
                    raise ValueError("El área debe ser un número positivo.")

                capacidad = int(entries["Capacidad"].get())
                if capacidad < 0:
                    raise ValueError("La capacidad no puede ser negativa.")

                duracion = float(entries["Duración Actividad (h)"].get())
                if duracion < 0:
                    raise ValueError("La duración de la actividad no puede ser negativa.")

                es_exterior = entries["Es Exterior"].get().strip().lower() == "sí"
                x=np.random.uniform(0, self.sistema.building.floor_width)
                y=np.random.uniform(0, self.sistema.building.floor_length)
                z=np.random.randint(0, self.sistema.building.floors)

                espacio = Espacio(
                    nombre,
                    area,
                    capacidad,
                    actividad_var.get(),
                    duracion,
                    x,
                    y,
                    z,
                    es_exterior
                )
                self.sistema.building.add_espacio(espacio)
                self.adjacency_checker.update_adjacency_matrices()
                self.actualizar_visualizacion()
                dialogo.destroy()
            except ValueError as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(dialogo, text="Guardar",
                   command=guardar_espacio).grid(row=row + 1, column=0, columnspan=2, pady=10)

    def mostrar_dialogo_dispositivo(self):
        if not self.sistema.building.espacios:
            messagebox.showwarning("Advertencia", "Primero debe crear al menos un espacio")
            return

        dialogo = tk.Toplevel(self.root)
        dialogo.title("Agregar Dispositivo de Iluminación")

        campos = {
            "Tipo": "",
            "Potencia (W)": "0.0",
            "Eficiencia (lm/W)": "0.0",
            "Temperatura Color (K)": "0.0"
        }

        row = 0
        entries = {}
        for campo, valor in campos.items():
            ttk.Label(dialogo, text=campo).grid(row=row, column=0, pady=5)
            entry = ttk.Entry(dialogo)
            entry.insert(0, valor)
            entry.grid(row=row, column=1, pady=5)
            entries[campo] = entry
            row += 1

        ttk.Label(dialogo, text="Espacio:").grid(row=row, column=0, pady=5)
        espacio_var = tk.StringVar()
        espacio_combo = ttk.Combobox(dialogo, textvariable=espacio_var)
        espacio_combo['values'] = [espacio.nombre for espacio in self.sistema.espacios]
        espacio_combo.grid(row=row, column=1, pady=5)

        def guardar_dispositivo():
            try:
                dispositivo = DispositivoIluminacion(
                    entries["Tipo"].get(),
                    float(entries["Potencia (W)"].get()),
                    float(entries["Eficiencia (lm/W)"].get()),
                    float(entries["Temperatura Color (K)"].get())
                )

                for espacio in self.sistema.espacios:
                    if espacio.nombre == espacio_var.get():
                        espacio.dispositivos_iluminacion.append(dispositivo)
                        break

                self.actualizar_visualizacion()
                dialogo.destroy()
            except ValueError:
                messagebox.showerror("Error", "Por favor, ingrese valores válidos")

        ttk.Button(dialogo, text="Guardar",
                   command=guardar_dispositivo).grid(row=row + 1, column=0, columnspan=2, pady=10)

    def mostrar_analisis(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para analizar")
            return

        hora_dia = int(self.simulador.hora_dia)
        clima = self.clima_var.get()
        resultados = self.sistema.analizar_iluminacion_global(CondicionesExternas(self.simulador.iluminancia_exterior, self.simulador.hora_dia, self.simulador.clima))

        ventana = tk.Toplevel(self.root)
        ventana.title("Análisis de Iluminación")

        texto = tk.Text(ventana, wrap=tk.WORD, width=60, height=20)
        texto.pack(padx=10, pady=10)

        for espacio in self.sistema.espacios:
            es_habitable = resultados[espacio.nombre]['habitable']
            recomendacion = self.recomendaciones.obtener_recomendacion(
                espacio.actividad_principal, 
                espacio.es_exterior, 
                clima,
                es_habitable
            )
            
            texto.insert(tk.END, f"\nEspacio: {espacio.nombre}\n")
            texto.insert(tk.END, f"Nivel de iluminación: {resultados[espacio.nombre]['nivel_iluminacion']:.2f} lux\n")
            texto.insert(tk.END, f"Lumens: {resultados[espacio.nombre]['lumens']:.2f} lm\n")
            texto.insert(tk.END, f"Uniformidad: {resultados[espacio.nombre]['uniformidad']:.2f}\n")
            texto.insert(tk.END, f"Deslumbramiento: {resultados[espacio.nombre]['deslumbramiento']:.2f}\n")
            texto.insert(tk.END, f"Rendimiento de color: {resultados[espacio.nombre]['rendimiento_color']:.2f}\n")
            texto.insert(tk.END, f"¿Es habitable? {'Sí' if es_habitable else 'No'}\n")
            texto.insert(tk.END, "Recomendaciones:\n")
            for rec in recomendacion:
                texto.insert(tk.END, f"- {rec}\n")

    def mostrar_optimizacion(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para optimizar")
            return

        optimizaciones = self.sistema.optimizar_consumo_energetico()

        ventana = tk.Toplevel(self.root)
        ventana.title("Optimización de Consumo Energético")

        texto = tk.Text(ventana, wrap=tk.WORD, width=60, height=20)
        texto.pack(padx=10, pady=10)

        for espacio, datos in optimizaciones.items():
            texto.insert(tk.END, f"\nEspacio: {espacio}\n")
            texto.insert(tk.END, f"Ajuste de Intensidad: Reducir la intensidad de la luz en {datos['ajuste_intensidad']:.2f}%\n")
            texto.insert(tk.END, f"Ahorro Estimado: {max(0, datos['ahorro_estimado']):.2f}%\n")

    def actualizar_visualizacion(self):
        self.fig.clear()
        for espacio in self.sistema.espacios:
            espacio.visualizador.visualizar_espacio_3d(self.sistema.espacios)
        self.canvas.draw()


# Simulación dinámica de condiciones externas
class SimuladorCondicionesExternas:
    def __init__(self, sistema):
        self.sistema = sistema
        self.hora_dia = 8  # Comienza a las 8 AM
        self.clima = "despejado"
        self.iluminancia_exterior = 10000  # en lux
        self.running = True

    def simular(self):
        while self.running:
            self.hora_dia = (self.hora_dia + 1) % 24
            if self.hora_dia < 6 or self.hora_dia > 18:
                self.iluminancia_exterior = 0  # Noche
            else:
                self.iluminancia_exterior = 10000  # Día
            time.sleep(10)  # Simula un minuto cada segundo

    def obtener_hora_actual(self) -> str:
        hora = self.hora_dia % 12
        if hora == 0:
            hora = 12
        am_pm = "AM" if self.hora_dia < 12 else "PM"
        return f"{hora:02d}:00 {am_pm}"

  # Refresh every second


    def iniciar_simulacion(self):
        threading.Thread(target=self.simular, daemon=True).start()


# Start with no predefined building
sistema = None  
interfaz = InterfazGrafica(sistema)  # GUI starts without a predefined building
interfaz.root.mainloop()

