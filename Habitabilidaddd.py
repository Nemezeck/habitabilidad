import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import math
import threading
import time
from typing import Dict, List, Tuple, Set


class CondicionesExternas:
    def __init__(self, iluminancia_exterior: float, hora_dia: int, clima: str):
        self.hora_dia = hora_dia
        self.clima = clima
        self.iluminancia_exterior = self.calcular_iluminancia()

    def calcular_iluminancia(self) -> float:
        """ Ajusta la iluminancia exterior basada en la hora del día y el clima. """
        # Valores base de iluminancia (lux) según la hora
        iluminancia_maxima = 100000  # Máximo a mediodía en un día despejado
        iluminancia_minima = 10      # Noche

        # Ajuste de iluminancia basado en la hora del día (Curva senoidal)
        if 6 <= self.hora_dia <= 18:  # De 6 AM a 6 PM
            factor_hora = math.sin(math.pi * (self.hora_dia - 6) / 12)  # 0 a 1
            iluminancia_base = iluminancia_minima + (iluminancia_maxima - iluminancia_minima) * factor_hora
        else:  # Noche
            iluminancia_base = iluminancia_minima

        # Ajuste por clima
        factores_climaticos = {
            "despejado": 1.0,
            "parcialmente nublado": 0.6,
            "nublado": 0.3,
            "lluvioso": 0.1
        }
        factor_clima = factores_climaticos.get(self.clima.lower(), 1.0)  # Default a 1.0 si el clima no es reconocido

        iluminancia_final = iluminancia_base * factor_clima
        return max(iluminancia_final, 0)  # Asegurar que nunca sea negativo
class Ventana:
    def __init__(self, area: float, factor_transmision_luz: float, orientacion: str):
        self.area = max(0, area)  # Evitar valores negativos
        self.factor_transmision_luz = max(0, min(1, factor_transmision_luz))  # Debe estar entre 0 y 1
        self.orientacion = orientacion
        self.apertura_persiana = 100  # porcentaje

    def calcular_aporte_luz_natural(self, condiciones_externas: CondicionesExternas) -> float:
        if self.area == 0 or self.factor_transmision_luz == 0 or condiciones_externas.iluminancia_exterior == 0:
            return 0  # Evitar cálculos innecesarios

        aporte_luz = (self.area * self.factor_transmision_luz *
                      condiciones_externas.iluminancia_exterior *
                      (self.apertura_persiana / 100))

        # Depuración
        print(f"Área: {self.area} m², Factor transmisión: {self.factor_transmision_luz}, "
              f"Iluminancia exterior: {condiciones_externas.iluminancia_exterior} lux, "
              f"Apertura persiana: {self.apertura_persiana}%, Aporte luz: {aporte_luz} lux")

        return aporte_luz

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
        self.area = max(area, 0.1)  # Evitar área cero
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
        self.nivel_iluminacion = 0.0  # Inicializar iluminación
        self.superficies = []
        self.coloracion = ColoracionEspacio(self)
        self.visualizador = VisualizadorEspacio(self)

    def calcular_iluminacion_total(self, hora_dia: int) -> float:
        self.nivel_iluminacion = PropagacionLuz.calcular_iluminacion_espacio(self, hora_dia)
        return self.nivel_iluminacion

    def calcular_lumens(self) -> float:
        return self.nivel_iluminacion * self.area

    def es_habitable(self) -> bool:
        thresholds = {
            "Reuniones": 300,
            "Estudio": 400,
            "Descanso": 200,
        }
        return self.nivel_iluminacion >= thresholds.get(self.actividad_principal, 0)


class Building:
    def __init__(self, nombre: str, floors: int, floor_width: float, floor_length: float):
        self.nombre = nombre
        self.floors = floors
        self.floor_width = floor_width
        self.floor_length = floor_length
        self.espacios = []

    def add_espacio(self, espacio: Espacio):
        # Verificar si el espacio está dentro de los límites del edificio
        if 0 <= espacio.x <= self.floor_width and 0 <= espacio.y <= self.floor_length and 0 <= espacio.z < self.floors:
            # Si no tiene ventanas y no es un espacio sin acceso a luz natural, agregamos una ventana predeterminada
            if not espacio.ventanas and not espacio.es_exterior:
                print(f"[Aviso] {espacio.nombre} no tiene ventanas. Se agregará una por defecto.")
                espacio.ventanas.append(Ventana(area=2, factor_transmision_luz=0.7, orientacion="sur"))

            self.espacios.append(espacio)
        else:
            raise ValueError("Las coordenadas del espacio están fuera de los límites del edificio.")


class PropagacionLuz:
    @staticmethod
    def calcular_iluminacion_punto(punto: Tuple[float, float, float], fuentes_luz: List, superficies: List) -> float:
        iluminacion_directa = PropagacionLuz._calcular_iluminacion_directa_sin_posicion(fuentes_luz)
        iluminacion_indirecta = PropagacionLuz._calcular_iluminacion_indirecta(punto, superficies)
        return iluminacion_directa + iluminacion_indirecta
    @staticmethod
    def _calcular_iluminacion_directa(punto: Tuple[float, float, float],
                                      fuente) -> float:
        distancia = 1.0  # En una implementación real, calcular la distancia
        return fuente.obtener_flujo_luminoso() / (4 * math.pi * distancia ** 2)

    @staticmethod
    def _calcular_iluminacion_indirecta(punto: Tuple[float, float, float], superficies: List) -> float:
        return 50  # Valor base simplificado
    @staticmethod
   
    def calcular_iluminacion_espacio(espacio: Espacio, hora_dia: int) -> float:
        # Calcular luz artificial solo si los dispositivos están encendidos
        iluminacion_artificial = sum(dispositivo.obtener_flujo_luminoso()
                                     for dispositivo in espacio.dispositivos_iluminacion
                                     if dispositivo.estado)  # Solo dispositivos encendidos

        # Calcular luz natural
        condiciones_externas = CondicionesExternas(100000, hora_dia, "despejado")  # Valores de ejemplo
        iluminacion_natural = sum(ventana.calcular_aporte_luz_natural(condiciones_externas)
                                  for ventana in espacio.ventanas)

        # Ajuste para exteriores
        if espacio.es_exterior:
            iluminacion_natural *= 1.5

        # Evitar división por cero en el área
        if espacio.area <= 0:
            print(f"Advertencia: El área de {espacio.nombre} es cero o negativa.")
            return 0.0

        # Suma de iluminación total dividida por el área del espacio
        iluminacion_total = (iluminacion_artificial + iluminacion_natural) / espacio.area

        # Asegurar que siempre haya un mínimo de luz
        return max(iluminacion_total, 0.1)


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
        self.building = building  # Renombramos de "edificio" a "building" para evitar errores


    def actualizar_iluminacion(self, iluminancia_exterior: float, hora_dia: int, clima: str):
        condiciones = CondicionesExternas(iluminancia_exterior, hora_dia, clima)
        for espacio in self.building.espacios:
            luz_natural_total = sum(ventana.calcular_aporte_luz_natural(condiciones) for ventana in espacio.ventanas)
            espacio.nivel_iluminacion = luz_natural_total
            print(f"[Actualización] {espacio.nombre} -> Luz Natural: {luz_natural_total:.2f} lux")
    @property
    def espacios(self):
        return self.building.espacios  # Ahora es una propiedad, no un método
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
                    "Instalar iluminación adicional para alcanzar los 200 lux mínimos requeridos.",
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

class InterfazGrafica:
    def __init__(self, sistema):
        self.sistema = sistema
        self.simulador = SimuladorCondicionesExternas(self.sistema)
         # Definir las variables aquí para que se mantengan actualizadas
        self.root = tk.Tk()
        self.root.title("Sistema de Habitabilidad")
        self.root.geometry("1280x720")
        self.configurar_interfaz()
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
        self.recomendaciones = Recomendaciones()
    def configurar_interfaz(self):
        self.frame_simulacion = ttk.Frame(self.root, padding="10")
        self.frame_simulacion.pack(fill=tk.BOTH, expand=True)

        ttk.Label(self.frame_simulacion, text="Hora del día:").pack()
        self.hora_var = tk.IntVar()  # Sin valor predeterminado
        ttk.Spinbox(self.frame_simulacion, from_=0, to=23, textvariable=self.hora_var).pack()

        ttk.Label(self.frame_simulacion, text="Clima:").pack()
        self.clima_var = tk.StringVar()  # Sin valor predeterminado
        ttk.Combobox(self.frame_simulacion, values=["despejado", "nublado", "lluvioso"], textvariable=self.clima_var).pack()

        ttk.Button(self.frame_simulacion, text="Actualizar Simulación", command=self.actualizar_simulacion).pack()


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
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        self.panel_simulacion = ttk.Frame(self.control_frame)
        self.panel_simulacion.pack(pady=10)
        self.crear_botones()
        ttk.Label(self.panel_simulacion, text="Hora del día:").grid(row=0, column=0, pady=5)
        # Ahora reutilizamos self.hora_var, no la volvemos a definir
        self.hora_spinbox = ttk.Spinbox(self.panel_simulacion, from_=0, to=23, textvariable=self.hora_var)
        self.hora_spinbox.grid(row=0, column=1, pady=5)
        
        ttk.Label(self.panel_simulacion, text="Clima:").grid(row=1, column=0, pady=5)
        # Reutilizamos self.clima_var
        self.clima_combobox = ttk.Combobox(self.panel_simulacion, values=["despejado", "nublado", "lluvioso"], textvariable=self.clima_var)
        self.clima_combobox.grid(row=1, column=1, pady=5)

        # Botón para actualizar manualmente
        self.actualizar_btn = ttk.Button(self.panel_simulacion, text="Actualizar Simulación", command=self.actualizar_simulacion)
        self.actualizar_btn.grid(row=2, columnspan=2, pady=10)
        
        # Añadimos botones para aplicar algoritmos de coloración
        
    def crear_botones(self):
        ttk.Button(self.control_frame, text="Coloración Greedy", command=self.aplicar_coloracion_greedy).pack(pady=5)
        ttk.Button(self.control_frame, text="Coloración Robusta", command=self.aplicar_coloracion_robusta).pack(pady=5)
        ttk.Button(self.control_frame, text="Vista 3D", command=self.mostrar_vista_3d).pack(pady=5)
        ttk.Button(self.control_frame, text="Vista 2D", command=self.mostrar_vista_2d).pack(pady=5)
        ttk.Button(self.control_frame, text="Control de Dispositivos", command=self.mostrar_control_dispositivos).pack(pady=5)
        ttk.Button(self.control_frame, text="Info Algoritmos", command=self.mostrar_informacion_algoritmos).pack(pady=5)
    def actualizar_simulacion(self):
         # Aquí aseguramos que estamos pasando los valores correctos
        nueva_hora = self.hora_var.get()
        nuevo_clima = self.clima_var.get()
        print(f"Actualizando a hora: {nueva_hora}, clima: {nuevo_clima}")
        
        # Llamamos a la función de actualización en el simulador con los valores capturados
        self.simulador.actualizar_condiciones(nueva_hora, nuevo_clima)
    

    def aplicar_coloracion_greedy(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para colorear")
            return

        espacio_actual = self.sistema.espacios[0]
        espacio_actual.coloracion.construir_grafo_adyacencia()
        coloracion = espacio_actual.coloracion.coloracion_greedy()
        self.visualizar_coloracion(coloracion, "Coloración Greedy")

    def aplicar_coloracion_robusta(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para colorear")
            return

        espacio_actual = self.sistema.espacios[0]
        espacio_actual.coloracion.construir_grafo_adyacencia()
        coloracion = espacio_actual.coloracion.coloracion_robusta(5)
        self.visualizar_coloracion(coloracion, "Coloración Robusta")

    def visualizar_coloracion(self, coloracion: Dict[int, int], tipo: str):
        ventana_resultados = tk.Toplevel(self.root)
        ventana_resultados.title(f"Resultados de {tipo}")

        texto = tk.Text(ventana_resultados, wrap=tk.WORD, width=60, height=20)
        texto.pack(padx=10, pady=10)

        for nodo, color in coloracion.items():
            texto.insert(tk.END, f"Zona {nodo}: Color {color}\n")

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        zonas = self.sistema.espacios[0].coloracion._dividir_espacio_en_zonas()

        for nodo, color in coloracion.items():
            zona = zonas[nodo]
            x, y = zona['centro']

        # Ensure color is an integer and within the valid range for tab20
        if isinstance(color, int):
            color_index = max(0, min(19, color))  # Clamp to the range [0, 19]
        else:
            print(f"Invalid color value for node {nodo}: {color}. Defaulting to 0.")
            color_index = 0  # Default to a valid color index

        ax.scatter(x, y, c=[plt.cm.tab20(color_index)], s=100)

        ax.set_title(f"Simulación de {tipo}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        self.canvas.draw()

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

    def mostrar_vista_3d(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para visualizar")
            return

        ventana_3d = tk.Toplevel(self.root)
        ventana_3d.title("Vista 3D del Edificio")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for espacio in self.sistema.espacios:
            color = 'green' if espacio.es_habitable() else 'red'
            ax.scatter(espacio.x, espacio.y, espacio.z, c=color, s=100)
            ax.text(espacio.x, espacio.y, espacio.z + 0.1, f"{espacio.nombre}", color='black', fontsize=8)

        ax.set_xlim(0, self.sistema.floor_width)
        ax.set_ylim(0, self.sistema.floor_length)
        ax.set_zlim(0, self.sistema.floors)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Floor (Z)')
        ax.set_title('Vista 3D del Edificio')

        canvas = FigureCanvasTkAgg(fig, master=ventana_3d)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def mostrar_vista_2d(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para visualizar")
            return

        plt.figure(figsize=(10, 8))
        for espacio in self.sistema.espacios:
            length = width = np.sqrt(espacio.area)
            color = 'green' if espacio.es_habitable() else 'red'
            plt.gca().add_patch(plt.Rectangle((0, 0), length, width, color=color, alpha=0.5))
            plt.text(length / 2, width / 2, espacio.nombre, horizontalalignment='center', verticalalignment='center')

        plt.xlim(0, max(np.sqrt(e.area) for e in self.sistema.espacios))
        plt.ylim(0, max(np.sqrt(e.area) for e in self.sistema.espacios))
        plt.title('Vista 2D de Todos los Espacios')
        plt.xlabel('Longitud (m)')
        plt.ylabel('Anchura (m)')
        plt.grid()
        plt.show()

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
        dispositivo_combo.grid(row=0, column=0, pady=3)

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

    def configurar_configuracion_tab(self):
        self.configuracion_frame = ttk.Frame(self.configuracion_tab, padding="10")
        self.configuracion_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Button(self.configuracion_frame, text="Agregar Espacio", command=self.mostrar_dialogo_espacio).pack(pady=5)
        ttk.Button(self.configuracion_frame, text="Agregar Dispositivo", command=self.mostrar_dialogo_dispositivo).pack(pady=5)

    def configurar_analisis_tab(self):
        self.analisis_frame = ttk.Frame(self.analisis_tab, padding="10")
        self.analisis_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(self.analisis_frame, text="Analizar Iluminación", command=self.mostrar_analisis).pack(pady=5)

    def configurar_optimizacion_tab(self):
        self.optimizacion_frame = ttk.Frame(self.optimizacion_tab, padding="10")
        self.optimizacion_frame.pack(fill=tk.BOTH, expand=True)
        ttk.Button(self.optimizacion_frame, text="Optimizar Consumo", command=self.mostrar_optimizacion).pack(pady=5)
        
    
    def aplicar_coloracion_greedy(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para colorear")
            return

        espacio_actual = self.sistema.espacios[0]
        espacio_actual.coloracion.construir_grafo_adyacencia()
        coloracion = espacio_actual.coloracion.coloracion_greedy()

        # Visualizar la simulación
        self.visualizar_coloracion(coloracion, "Coloración Greedy")

    def aplicar_coloracion_robusta(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para colorear")
            return

        espacio_actual = self.sistema.espacios[0]
        espacio_actual.coloracion.construir_grafo_adyacencia()
        coloracion = espacio_actual.coloracion.coloracion_robusta(5)

        # Visualizar la simulación
        self.visualizar_coloracion(coloracion, "Coloración Robusta")

    def visualizar_coloracion(self, coloracion: Dict[int, Set[int]], tipo: str):
            ventana_resultados = tk.Toplevel(self.root)
            ventana_resultados.title(f"Resultados de {tipo}")

            texto = tk.Text(ventana_resultados, wrap=tk.WORD, width=60, height=20)
            texto.pack(padx=10, pady=10)

            for nodo, colores in coloracion.items():
                texto.insert(tk.END, f"Zona {nodo}: Colores {colores}\n")

            self.fig.clear()
            ax = self.fig.add_subplot(111)
            zonas = self.sistema.espacios[0].coloracion._dividir_espacio_en_zonas()

            for nodo, colores in coloracion.items():
                zona = zonas[nodo]
                x, y = zona['centro']

        # Choose the first color from the set (or implement your own logic)
                color_index = next(iter(colores)) if colores else 0  # Default to 0 if no colors available

        # Ensure color is within the valid range for tab20
                color_index = max(0, min(19, color_index))  # Clamp to the range [0, 19]

                ax.scatter(x, y, c=[plt.cm.tab20(color_index)], s=100)

                ax.set_title(f"Simulación de {tipo}")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                self.canvas.draw()

    def mostrar_vista_3d(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para visualizar")
            return

        ventana_3d = tk.Toplevel(self.root)
        ventana_3d.title("Vista 3D de Todos los Espacios")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot each Espacio as a single point
        for espacio in self.sistema.espacios:
            x, y, z = espacio.x, espacio.y, espacio.z  # Correct reference
            color = 'green' if espacio.es_habitable() else 'red'
            ax.scatter(x, y, z, c=color, s=50)
            ax.text(x, y, z + 0.2, espacio.nombre, color='black', fontsize=8)

        # Determine the limits dynamically
        max_x = max(e.x for e in self.sistema.espacios) + 1
        max_y = max(e.y for e in self.sistema.espacios) + 1
        max_z = max(e.z for e in self.sistema.espacios) + 1

        # Set axis limits
        ax.set_xlim(0, max_x)
        ax.set_ylim(0, max_y)
        ax.set_zlim(0, max_z)

        # Set axis labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # **Set grid spacing to 1 unit**
        ax.set_xticks(np.arange(0, max_x + 1, 1))
        ax.set_yticks(np.arange(0, max_y + 1, 1))
        ax.set_zticks(np.arange(0, max_z + 1, 1))

        # Set a title
        ax.set_title('Vista 3D de Todos los Espacios')

        # Display the plot in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=ventana_3d)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)




    def mostrar_vista_2d(self):
        if not self.sistema.espacios:
            messagebox.showwarning("Advertencia", "No hay espacios para visualizar")
            return

        plt.figure(figsize=(10, 8))
        for espacio in self.sistema.espacios:
            length = width = np.sqrt(espacio.area)
            color = 'green' if espacio.es_habitable() else 'red'
            plt.gca().add_patch(plt.Rectangle((0, 0), length, width, color=color, alpha=0.5))
            plt.text(length / 2, width / 2, espacio.nombre, horizontalalignment='center', verticalalignment='center')

        plt.xlim(0, max(np.sqrt(e.area) for e in self.sistema.espacios))
        plt.ylim(0, max(np.sqrt(e.area) for e in self.sistema.espacios))
        plt.title('Vista 2D de Todos los Espacios')
        plt.xlabel('Longitud (m)')
        plt.ylabel('Anchura (m)')
        plt.grid()
        plt.show()

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

                espacio = Espacio(
                    nombre,
                    area,
                    capacidad,
                    actividad_var.get(),
                    duracion,
                    es_exterior
                )
                self.sistema.agregar_espacio(espacio)
                self.actualizar_visualizacion()
                dialogo.destroy()
            except ValueError as e:
                messagebox.showerror("Error", str(e))

        ttk.Button(dialogo, text="Guardar",
                   command=guardar_espacio).grid(row=row + 1, column=0, columnspan=2, pady=10)

    def mostrar_dialogo_dispositivo(self):
        if not self.sistema.espacios:
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

        hora_dia = int(self.hora_var.get())
        clima = self.clima_var.get()
        resultados = self.sistema.analizar_iluminacion_global(hora_dia)

        ventana = tk.Toplevel(self.root)
        ventana.title("Análisis de Iluminación")

        texto = tk.Text(ventana, wrap=tk.WORD, width=60, height=20)
        texto.pack(padx=10, pady=10)

        for espacio in self.sistema.espacios:
            es_habitable = resultados[espacio.nombre]['habitable']
            clima = self.clima_var.get()  # Obtener el clima seleccionado en la interfaz
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
            texto.insert(tk.END, f"Ajuste de Intensidad: {datos['ajuste_intensidad']:.2f}%\n")
            texto.insert(tk.END, f"Ahorro Estimado: {datos['ahorro_estimado']:.2f}%\n")

    def actualizar_visualizacion(self):
        self.fig.clear()
        for espacio in self.sistema.espacios:
            espacio.visualizador.visualizar_espacio_3d(self.sistema.espacios)
        self.canvas.draw()


# Simulación dinámica de condiciones externas
class SimuladorCondicionesExternas:
    def __init__(self, sistema):
        self.sistema = sistema
        self.hora_dia = 0  # Comienza a las 8 AM
        self.clima = "nublado"
        self.iluminancia_exterior = self.calcular_iluminancia()
        self.running = True

    def actualizar_condiciones(self, nueva_hora, nuevo_clima):
        """ Actualiza las condiciones en la simulación con los nuevos valores de hora y clima. """
        self.hora_dia = nueva_hora
        self.clima = nuevo_clima
        print(f"[Actualización manual] Hora: {self.hora_dia}, Clima: {self.clima}")
        
        # Calcular la nueva iluminancia y actualizar el sistema
        self.iluminancia_exterior = self.calcular_iluminancia()
        self.sistema.actualizar_iluminacion(self.iluminancia_exterior, self.hora_dia, self.clima)

      

    def calcular_iluminancia(self) -> float:
        """ Ajusta la iluminancia exterior basada en la hora del día y el clima. """
        iluminancia_maxima = 100000  # Lux máximo a mediodía despejado
        iluminancia_minima = 10      # Lux mínimo en la noche

        if 6 <= self.hora_dia <= 18:  # Día (6 AM a 6 PM)
            factor_hora = math.sin(math.pi * (self.hora_dia - 6) / 12)  # 0 a 1
            iluminancia_base = iluminancia_minima + (iluminancia_maxima - iluminancia_minima) * factor_hora
        else:  # Noche
            iluminancia_base = iluminancia_minima

        factores_climaticos = {
            "despejado": 1.0,
            "parcialmente nublado": 0.6,
            "nublado": 0.3,
            "lluvioso": 0.1
        }
        factor_clima = factores_climaticos.get(self.clima.lower(), 1.0)

        return max(iluminancia_base * factor_clima, 0)

    def simular(self):
        """ Simula la variación de iluminación cada minuto automáticamente. """
        while self.running:
            self.hora_dia = (self.hora_dia + 1) % 24  # Avanza la hora automáticamente
            self.iluminancia_exterior = self.calcular_iluminancia()
            print(f"[Simulación] Hora: {self.hora_dia}:00, Clima: {self.clima}, Iluminancia exterior: {self.iluminancia_exterior:.2f} lux")

            # Notificar al sistema de iluminación
            self.sistema.actualizar_iluminacion(self.iluminancia_exterior, self.hora_dia, self.clima)

            time.sleep(5)  # Simula 1 hora cada 5 segundos (para pruebas rápidas)

    def iniciar_simulacion(self):
        hilo = threading.Thread(target=self.simular, daemon=True)
        hilo.start()

# Inicialización del sistema y la interfaz gráfica
building = Building("Edificio Central", floors=3, floor_width=20, floor_length=20)

sala_reuniones = Espacio("Sala Reuniones", area=25, capacidad_ocupacion=5, 
                         actividad_principal="Reuniones", duracion_actividad=2, x=5, y=5, z=0)

sala_reuniones.ventanas.append(Ventana(area=2, factor_transmision_luz=0.7, orientacion="sur"))

building.add_espacio(sala_reuniones)

biblioteca = Espacio("Biblioteca", area=30, capacidad_ocupacion=8, 
                     actividad_principal="Estudio", duracion_actividad=4, x=10, y=15, z=1)

biblioteca.ventanas.append(Ventana(area=3, factor_transmision_luz=0.75, orientacion="este"))

dormitorio = Espacio("Dormitorio", area=20, capacidad_ocupacion=2, 
                     actividad_principal="Descanso", duracion_actividad=8, x=7, y=3, z=2)

dormitorio.ventanas.append(Ventana(area=2, factor_transmision_luz=0.6, orientacion="oeste"))

building.add_espacio(biblioteca)
building.add_espacio(dormitorio)

sistema = SistemaIluminacion(building)
interfaz = InterfazGrafica(sistema)
interfaz.root.mainloop()

