import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import threading
import time
from typing import Dict, List, Tuple, Set
import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Button
from matplotlib.figure import Figure


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
        self.y = y  # Vertical position
        self.z = z  # Floor number
        self.es_exterior = es_exterior
        self.ventanas = []
        self.dispositivos_iluminacion = []
        self.ocupantes = []
        self.nivel_iluminacion = 0
        self.superficies = []
        self.coloracion = ColoracionEspacio(self)
        self.visualizador = VisualizadorEspacio(self)
        self.offset_pct = 0


    def calcular_iluminacion_total(self, hora_dia: int) -> float:
        self.nivel_iluminacion = PropagacionLuz.calcular_iluminacion_espacio(self, hora_dia)
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
    def calcular_iluminacion_espacio(espacio: Espacio, hora_dia: int) -> float:
        iluminacion_artificial = sum(dispositivo.obtener_flujo_luminoso()
                                     for dispositivo in espacio.dispositivos_iluminacion)
        iluminacion_natural = sum(ventana.calcular_aporte_luz_natural(
            CondicionesExternas(10000, hora_dia, "despejado"))  # Valores ejemplo
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

    def analizar_iluminacion_global(self, hora_dia: int) -> Dict:
        if not self.espacios:
            print("No hay espacios en el edificio.")
            return {}

        resultados = {}
        for espacio in self.espacios:
            resultados[espacio.nombre] = {
                'nivel_iluminacion': espacio.calcular_iluminacion_total(hora_dia),
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
                'ajuste_intensidad': np.random.uniform(60, 100),
                'ahorro_estimado': np.random.uniform(5, 30)
            }
        return optimizaciones



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
                }
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
                }
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
                }
            }
        }

    def obtener_recomendacion(self, actividad_principal: str, es_exterior: bool, clima: str) -> List[str]:
        tipo_ubicacion = "exterior" if es_exterior else "interior"
        return self.recomendaciones.get(actividad_principal, {}).get(tipo_ubicacion, {}).get(clima, ["No hay recomendaciones disponibles."])
    
class AdjacencyChecker:
    def __init__(self, spaces, threshold=5):
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
    
class InterfazGrafica:
    def __init__(self, building):
        self.root = tk.Tk()
        self.root.title("Building Adjacency Visualization")
        self.building = building
        self.adjacency_checker = AdjacencyChecker(building.espacios)

        # Initialize figure before using it
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.axs[1] = self.fig.add_subplot(122, projection='3d')

        # Now it's safe to create the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        self.btn_plot = Button(self.root, text="Plot Graph", command=self.plot_graph)
        self.btn_plot.pack()

    def plot_graph(self):
        self.axs[0].cla()
        self.axs[1].cla()
        self.plot_colored_graph()
        self.plot_3d_graph()
        self.canvas.draw()  # Ensure canvas updates
    
    def greedy_coloring(self):
        n = len(self.building.espacios)
        colors = [-1] * n  
        
        for i in range(n):
            adjacent_colors = set()
            for j in range(n):
                if self.adjacency_checker.adjacency_matrix_2d[i][j] == 1 and colors[j] != -1:
                    adjacent_colors.add(colors[j])

        
            color = 0
            while color in adjacent_colors:
                color += 1
            self.building.espacios[i].calcular_iluminacion_total(12)
            if self.building.espacios[i].es_habitable():
                
                colors[i] = 'g'  
            else:
                offset_pct = (self.building.espacios[i].nivel_iluminacion / self.get_threshold(self.building.espacios[i].actividad_principal)) * 100
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

    def plot_2d_graph(self):
        ax = self.axs[0]
        ax.set_title("2D Adjacency (Per Floor)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        for i, espacio in enumerate(self.building.espacios):
            color = 'r'  
            if espacio.es_habitable():
                color = 'g'  
            else:
                if abs(espacio.offset_pct - 100) < 10:  
                    color = 'y'  
            
            ax.scatter(espacio.x, espacio.y, color=color, label=espacio.nombre if i == 0 else "")
            
            for j, is_adj in enumerate(self.adjacency_checker.adjacency_matrix_2d[i]):
                if is_adj:
                    ax.plot([espacio.x, self.building.espacios[j].x],
                            [espacio.y, self.building.espacios[j].y], 'r-')

    def plot_colored_graph(self):
        ax = self.axs[0]
        ax.set_title("2D Adjacency (Per Floor)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        colors = self.greedy_coloring()  # Get the greedy coloring result

        for i, espacio in enumerate(self.building.espacios):
            ax.scatter(espacio.x, espacio.y, color=colors[i], label=espacio.nombre if i == 0 else "")
            for j, is_adj in enumerate(self.adjacency_checker.adjacency_matrix_2d[i]):
                if is_adj:
                    ax.plot([espacio.x, self.building.espacios[j].x],
                            [espacio.y, self.building.espacios[j].y], 'r-')

    def plot_3d_graph(self):
        ax = self.axs[1]
        ax.set_title("3D Adjacency Graph")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        for i, espacio in enumerate(self.building.espacios):
            ax.scatter(espacio.x, espacio.y, espacio.z, color='g', label=espacio.nombre if i == 0 else "")
            for j, is_adj in enumerate(self.adjacency_checker.adjacency_matrix_3d[i]):
                if is_adj:
                    ax.plot([espacio.x, self.building.espacios[j].x],
                            [espacio.y, self.building.espacios[j].y],
                            [espacio.z, self.building.espacios[j].z], 'r-')

        plt.legend()




building = Building("Edificio Central", floors=3, floor_width=20, floor_length=20)

building.add_espacio(Espacio("Sala Reuniones", area=25, capacidad_ocupacion=5, actividad_principal="Reuniones", duracion_actividad=2, x=5, y=5, z=1))
building.add_espacio(Espacio("Biblioteca", area=30, capacidad_ocupacion=8, actividad_principal="Estudio", duracion_actividad=4, x=10, y=15, z=1))
building.add_espacio(Espacio("Dormitorio", area=20, capacidad_ocupacion=2, actividad_principal="Descanso", duracion_actividad=8, x=3, y=3, z=1))
building.add_espacio(Espacio("Dormitorio", area=20, capacidad_ocupacion=2, actividad_principal="Estudio", duracion_actividad=8, x=15, y=15, z=1))
building.add_espacio(Espacio("Dormitorio", area=20, capacidad_ocupacion=2, actividad_principal="Reuniones", duracion_actividad=8, x=6, y=9, z=1))
building.add_espacio(Espacio("Dormitorio", area=20, capacidad_ocupacion=2, actividad_principal="Estudio", duracion_actividad=8, x=2, y=1, z=1))
building.add_espacio(Espacio("Dormitorio", area=20, capacidad_ocupacion=2, actividad_principal="Descanso", duracion_actividad=8, x=10, y=10, z=1))

espacio = building.espacios[0]
dispositivo = DispositivoIluminacion(tipo="LED", potencia=3500, eficiencia_luminica=2, temperatura_color=4000)
dispositivo.encender()
espacio.dispositivos_iluminacion.append(dispositivo)
print(dispositivo.obtener_flujo_luminoso())

sistema = SistemaIluminacion(building)
interfaz = InterfazGrafica(building)
interfaz.root.mainloop()