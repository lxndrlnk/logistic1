from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import folium
from itertools import cycle
from collections import defaultdict
import csv
from geopy.distance import geodesic
from datetime import datetime
import configparser
import sys
import copy
from math import cos, pi 
import io
import streamlit as st
from streamlit_folium import folium_static  
import tempfile
import zipfile
import os
import chardet

class LoadingPoint: 
    def __init__(self, id, latitude = 56.919021, longitude = 60.764404, node=703608117, cluster=None, individual_orders=None): # это координаты и номер узла графа склада 
        self.id = id
        self.latitude = latitude
        self.longitude = longitude
        self.node = node
        self.in_ekaterinburg = True
        self.cluster = cluster
        self.individual_orders = individual_orders
        
class DeliveryPoint:
    def __init__(self, id, latitude, longitude, weight=0, volume=0, pallets=0, individual_orders=None):
        self.id = [id] if not isinstance(id, list) else id
        self.latitude = latitude
        self.longitude = longitude
        self.weight = weight  # Общий вес всех заказов
        self.volume = volume  # Общий объем всех заказов
        self.pallets = pallets  # Общее количество паллет всех заказов
        self.individual_orders = individual_orders if individual_orders is not None else []
        # остальные атрибуты
        self.cluster = None
        self.node = None
        self.x = None
        self.y = None
        self.delivery_sequence = None
        self.in_ekaterinburg = None  # Добавленный атрибут для указания местоположения

class DeliveryPointsManager:
    def __init__(self):
        self.delivery_points = []

    def add_from_dataframe(self, df):
        grouped_points = {}
        for _, row in df.iterrows():
            coord_key = (row['lat'], row['lon'])
            individual_order = {
                'id': row['document'],
                'weight': row['doc_weight'],
                'volume': row['doc_vol'],
                'pallets': row['doc_palletes']
            }

            if coord_key not in grouped_points:
                grouped_points[coord_key] = {
                    'ids': [row['document']],
                    'weight': row['doc_weight'],
                    'volume': row['doc_vol'],
                    'pallets': row['doc_palletes'],
                    'individual_orders': [individual_order]
                }
            else:
                grouped_point = grouped_points[coord_key]
                grouped_point['ids'].append(row['document'])
                grouped_point['weight'] += row['doc_weight']
                grouped_point['volume'] += row['doc_vol']
                grouped_point['pallets'] += row['doc_palletes']
                grouped_point['individual_orders'].append(individual_order)

        for coord_key, data in grouped_points.items():
            self.delivery_points.append(
                DeliveryPoint(
                    id=data['ids'],
                    latitude=coord_key[0],
                    longitude=coord_key[1],
                    weight=data['weight'],
                    volume=data['volume'],
                    pallets=data['pallets'],
                    individual_orders=data['individual_orders']
                )
            )

class EkaterinburgDeliveryProcessor:
    def __init__(self, delivery_points_manager):
        self.points = delivery_points_manager
        self.G_proj = None
        self.distance_matrix = None

    def load_graph(self, RADIUS):
        # Определение центра Екатеринбурга и радиуса для загрузки графа
        ekaterinburg_center = (56.838011, 60.597465)
        radius_km = RADIUS
        st.write("Загрузка дорожного графа домашнего региона")
        # Загрузка и проекция графа
        self.G = ox.graph_from_point(ekaterinburg_center, dist=radius_km * 1000, network_type='drive')
        st.success(f"Граф загружен. Общее количество узлов в графе: {self.G.number_of_nodes()}")
        return self.G

    def assign_points_to_graph(self):
        # Назначение каждой точке доставки ближайшего узла графа и определение, находится ли она в Екатеринбурге
        for point in self.points.delivery_points:
            distance = geodesic((point.latitude, point.longitude), (56.838011, 60.597465)).km
            if distance <= 20:
                point.node = ox.distance.nearest_nodes(self.G, point.longitude, point.latitude)
                point.in_ekaterinburg = True
            else:
                point.in_ekaterinburg = False

    def calculate_distance_matrix(self):
        warehouse_lat = 56.919021
        warehouse_lon = 60.764404

        # Определение узла склада
        warehouse_node = ox.distance.nearest_nodes(self.G, warehouse_lon, warehouse_lat)

        # Фильтрация точек, находящихся в Екатеринбурге
        ekaterinburg_points = [point for point in self.points.delivery_points if point.in_ekaterinburg]

        # Добавляем узел склада в список уникальных узлов
        unique_nodes = set(point.node for point in ekaterinburg_points)
        unique_nodes.add(warehouse_node)
        self.distance_matrix = {node: {} for node in unique_nodes}

        # Расчет матрицы расстояний
        for i, node1 in enumerate(unique_nodes):
            #print(f"Processing node {i} of {len(unique_nodes)}")
            for j, node2 in enumerate(unique_nodes):
                if node1 == node2:
                    self.distance_matrix[node1][node2] = 0
                elif j > i:  # Вычисляем расстояние только один раз для каждой пары
                    try:
                        distance = nx.shortest_path_length(self.G, node1, node2, weight='length')
                        #print(distance, node1, node2)
                    except nx.NetworkXNoPath:
                        distance = float('inf')
                        st.write('Не найдено расстояние между', node1, node2)

                    self.distance_matrix[node1][node2] = distance
                    self.distance_matrix[node2][node1] = distance  # Зеркальное копирование для обеих комбинаций

    def cluster_delivery_points(self, num_clusters, cluster_offset):
        # Фильтрация точек, находящихся в Екатеринбурге
        ekaterinburg_points = [point for point in self.points.delivery_points if point.in_ekaterinburg]

        # Извлечение координат для кластеризации
        coordinates = np.array([(point.latitude, point.longitude) for point in ekaterinburg_points])

        # Кластеризация с использованием KMeans
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(coordinates)

        # Назначение меток кластеров точкам доставки
        for point, label in zip(ekaterinburg_points, kmeans.labels_):
            point.cluster = label + cluster_offset

   
          

class OutsideCityDeliveryProcessor:
    def __init__(self, delivery_points_manager):
        self.points = delivery_points_manager
        self.cluster_graphs = {}  # Словарь для хранения графов каждого кластера
        self.distance_matrices = {}  # Словарь для хранения матриц расстояний

    def cluster_outside_delivery_points(self, num_clusters):
        # Фильтрация точек, находящихся вне Екатеринбурга
        outside_points = [point for point in self.points.delivery_points if not point.in_ekaterinburg]

        # Извлечение координат для кластеризации
        coordinates = np.array([(point.latitude, point.longitude) for point in outside_points])

        # Кластеризация с использованием KMeans
        kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=0).fit(coordinates)

        # Назначение меток кластеров точкам доставки с учетом смещения
        for point, label in zip(outside_points, kmeans.labels_):
            point.cluster = label
        self.num_outside_clusters = num_clusters

    def load_cluster_graphs(self):
        # Загрузка и сохранение графов для каждого кластера
        for label in range(0, len(set(p.cluster for p in self.points.delivery_points if not p.in_ekaterinburg))):
            cluster_points = [point for point in self.points.delivery_points if point.cluster == label]
            if cluster_points:
                lats, lngs = zip(*[(point.latitude, point.longitude) for point in cluster_points])
                north, south, east, west = max(lats), min(lats), max(lngs), min(lngs)
                bbox_correction = 0.5
                G = ox.graph_from_bbox(north + bbox_correction, south - bbox_correction, 
                                       east + bbox_correction, west - bbox_correction, 
                                       network_type='drive', simplify=True)

                
                # Получаем количество узлов в графе
                num_nodes = G.number_of_nodes()
                
                self.cluster_graphs[label] = G
                  
                # Выводим информацию о загруженном графе
                st.success(f" Граф для кластера № {label} загружен. Общее количество узлов в графе: {num_nodes}")

    def assign_points_to_graph(self):
        for cluster_label, G in self.cluster_graphs.items():
            for point in [p for p in self.points.delivery_points if p.cluster == cluster_label]:
                point.node = ox.distance.nearest_nodes(G, point.longitude, point.latitude)

    def calculate_distance_matrices(self):
        for cluster_label, G in self.cluster_graphs.items():
            cluster_points = [point for point in self.points.delivery_points if point.cluster == cluster_label]
            unique_nodes = set(point.node for point in cluster_points)
            distance_matrix = {node: {} for node in unique_nodes}

            for i, node1 in enumerate(unique_nodes):
                #print(f"Processing node {i} of {len(unique_nodes)}")
                for j, node2 in enumerate(unique_nodes):
                    if node1 == node2:
                        distance_matrix[node1][node2] = 0
                    elif j > i:  # Вычисляем расстояние только один раз для каждой пары
                        try:
                            distance = nx.shortest_path_length(G, node1, node2, weight='length')
                        except nx.NetworkXNoPath:
                            distance = float('inf')
                            st.write('Не найдено расстояние между', node1, node2)
                        distance_matrix[node1][node2] = distance
                        distance_matrix[node2][node1] = distance
            self.distance_matrices[cluster_label] = distance_matrix
            #print(cluster_label)

    
class Vehicle:
    def __init__(self, id, capacity, volume, max_pallets):
        self.id = id
        self.capacity = capacity
        self.volume = volume
        self.max_pallets = max_pallets
        self.current_load = {'weight': 0, 'volume': 0, 'pallets': 0}
        self.orders = []  # Список объектов DeliveryPoint
        self.cluster_id = None  # ID кластера, к которому принадлежат заказы
        self.route_info = {}  # Словарь для хранения информации о маршруте
        self.is_used = False  # атрибут для отслеживания использования автомобиля
        self.distance_matrix = None
        self.closest_order_distance = 0



    def add_order(self, order, cluster_id):
        # Проверка, может ли автомобиль вместить заказ
        if self.can_accommodate_order(order):
            self.orders.append(order)
            self.current_load['weight'] += order.weight
            self.current_load['volume'] += order.volume
            self.current_load['pallets'] += order.pallets
            self.is_used = True  # флаг использования автомобиля
            # Устанавливаем cluster_id, если он еще не был установлен
            if self.cluster_id is None:
                self.cluster_id = cluster_id
            return True
        return False
    
    def can_accommodate_order(self, delivery_point):
        # Проверяем, может ли транспортное средство вместить заказ
        new_weight = self.current_load['weight'] + delivery_point.weight
        new_volume = self.current_load['volume'] + delivery_point.volume
        new_pallets = self.current_load['pallets'] + delivery_point.pallets
        return (new_weight <= self.capacity and 
                new_volume <= self.volume and 
                new_pallets <= self.max_pallets)
    
class VehiclesManager:
    def __init__(self):
        self.vehicles = []
        self.loading_order = []  # Список для сохранения порядка загрузки

    def add_from_dataframe(self, df):
        """
        Создает объекты Vehicle из DataFrame и добавляет их в список.
        """
        for _, row in df.iterrows():
            vehicle = Vehicle(
                id=row['truck'],
                capacity=row['truck_weight'],
                volume=row['truck_volume'],
                max_pallets=row['truck_palletes']
            )
            self.vehicles.append(vehicle)

    def load_vehicles_with_orders(self, delivery_points_manager, randomize=False, max_delivery_points_per_route_city=15, max_delivery_points_per_route_outside=150, log_file=False):

        if randomize:
            np.random.shuffle(self.vehicles)
        all_orders_loaded = True
        # Группировка точек доставки по кластерам с максимальным весом доставки в кластере
        clustered_points = {}
        cluster_max_weight = {}
        for point in delivery_points_manager.delivery_points:
            if point.cluster not in clustered_points:
                clustered_points[point.cluster] = []
                cluster_max_weight[point.cluster] = 0
            clustered_points[point.cluster].append(point)
            cluster_max_weight[point.cluster] = max(cluster_max_weight[point.cluster], point.weight)

        # Сортировка кластеров по максимальному весу в убывающем порядке
        sorted_clusters = sorted(clustered_points.keys(), key=lambda k: cluster_max_weight[k], reverse=True)
        for cluster_id in sorted_clusters:
            points = clustered_points[cluster_id]
            is_city_cluster = any(point.in_ekaterinburg for point in points)
            max_delivery_points_per_route = max_delivery_points_per_route_city if is_city_cluster else max_delivery_points_per_route_outside
            sorted_points = sorted(points, key=lambda x: x.weight, reverse=True)

            # Первый проход: пытаемся загрузить группированные точки доставки
            for vehicle in self.vehicles:
                if vehicle.is_used:
                    continue
                remaining_points = sorted_points.copy()
                for point in sorted_points:
                    if len(vehicle.orders) >= max_delivery_points_per_route:
                        break
                    if vehicle.add_order(point, cluster_id):
                        remaining_points.remove(point)
                sorted_points = remaining_points
                if not sorted_points:
                    break
            # Второй проход: обработка индивидуальных заказов
            if sorted_points:
                for point in sorted_points:
                    #if len(point.individual_orders) > 1:
                        #print(f"Агрегированный заказ {point.id}, Вес: {point.weight}, Объем: {point.volume}, Паллеты: {point.pallets} не был загружен и будет обработан индивидуально.")
                    if len(point.individual_orders) == 1:
                        st.write(f"Индивидуальный заказ {point.id[0]}, Вес: {point.weight}, Объем: {point.volume}, Паллеты: {point.pallets} не был загружен и не будет повторно рассматриваться.")
                        all_orders_loaded = False
                        return all_orders_loaded 
                    
            all_individual_orders = []
            for point in sorted_points:
                all_individual_orders.extend(point.individual_orders)

            remaining_individual_orders = all_individual_orders.copy()
            for alt_vehicle in self.vehicles:
                if alt_vehicle.is_used: 
                    continue

                for individual_order_info in all_individual_orders:
                    individual_order = DeliveryPoint(
                        id=individual_order_info['id'],
                        latitude=point.latitude, 
                        longitude=point.longitude, 
                        weight=individual_order_info['weight'],
                        volume=individual_order_info['volume'],
                        pallets=individual_order_info['pallets'],
                        individual_orders=[individual_order_info]  # Список, содержащий только данный индивидуальный заказ
                    )


                    # дополнительные атрибуты после создания объекта
                    individual_order.cluster = point.cluster
                    individual_order.node = point.node
                    individual_order.x = point.x
                    individual_order.y = point.y
                    individual_order.delivery_sequence = point.delivery_sequence
                    individual_order.in_ekaterinburg = point.in_ekaterinburg


                    if alt_vehicle.add_order(individual_order, cluster_id):
                        remaining_individual_orders.remove(individual_order_info)
                        #print(f"Индивидуальный заказ {individual_order_info['id']} был загружен.")

                all_individual_orders = remaining_individual_orders
                if not all_individual_orders:
                    break
            if all_individual_orders:
                for individual_order_info in all_individual_orders:
                    st.write(f"Индивидуальный заказ {individual_order_info['id']} не был загружен.")
                all_orders_loaded = False
                return all_orders_loaded  

        if all_orders_loaded and log_file:
            with open('loading_log.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Номер авто', 'Вес авто', 'Объем авто', 'Паллеты авто', 'Суммарная загрузка вес', 'Суммарная загрузка объем', 'Суммарная загрузка паллеты', 'Номер заказа', 'Вес заказа', 'Объем заказа', 'Паллеты заказа', 'Координаты заказа'])

                for vehicle in self.vehicles:
                    if vehicle.orders:
                        sum_weight = sum(order.weight for order in vehicle.orders)
                        sum_volume = sum(order.volume for order in vehicle.orders)
                        sum_pallets = sum(order.pallets for order in vehicle.orders)

                        for order in vehicle.orders:
                            writer.writerow([vehicle.id, vehicle.capacity, vehicle.volume, vehicle.max_pallets, sum_weight, sum_volume, sum_pallets, order.id, order.weight, order.volume, order.pallets, (order.latitude, order.longitude)])
        
        return all_orders_loaded 


    def add_warehouse_to_route(self, G):
        # Координаты склада
        warehouse_lat = 56.919021
        warehouse_lon = 60.764404

        # Находим ближайший узел к координатам склада
        warehouse_node = ox.distance.nearest_nodes(G, warehouse_lon, warehouse_lat)
        # Создаем точку загрузки для склада
        warehouse_point = LoadingPoint(id=0, latitude=warehouse_lat, longitude=warehouse_lon, node=warehouse_node)

        for vehicle in self.vehicles:
            # Проверяем, есть ли загрузка у автомобиля и все ли заказы находятся в городе
            if len(vehicle.orders) > 0 and all(order.in_ekaterinburg for order in vehicle.orders):
                vehicle.orders.insert(0, warehouse_point)  # Добавляем склад в начало
                vehicle.orders.append(warehouse_point)     # Добавляем склад в конец
            
 
    def adjust_warehouse_order_for_outside_routes(self, outside_processor):
        warehouse_lat = 56.919021
        warehouse_lon = 60.764404

        for vehicle in self.vehicles:
            if vehicle.orders and all(order.in_ekaterinburg == False for order in vehicle.orders):
                cluster_label = vehicle.orders[0].cluster
                G = outside_processor.cluster_graphs.get(cluster_label)

                if G is not None:
                    warehouse_node = ox.distance.nearest_nodes(G, warehouse_lon, warehouse_lat)
                    closest_order = min(vehicle.orders, key=lambda order: nx.shortest_path_length(G, order.node, warehouse_node, weight='length') if order.node else float('inf'))

                    distance_to_closest_order = nx.shortest_path_length(G, closest_order.node, warehouse_node, weight='length')
                    vehicle.closest_order_distance = distance_to_closest_order

                    vehicle.orders.remove(closest_order)
                    vehicle.orders.insert(0, closest_order)
                    vehicle.orders.append(closest_order)


    def plot_routes_on_map(self):
        """
        Отображает маршруты транспортных средств на карте.
        """
        map_center = [56.919021, 60.764404]
        m = folium.Map(location=map_center, zoom_start=10)

        # Цвета для маршрутов
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'black', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        color_iter = cycle(colors)

        for vehicle in self.vehicles:
            if 'route_gdf' in vehicle.route_info:
                route_color = next(color_iter)
                route_layer = folium.FeatureGroup(name=f"Маршрут {vehicle.id}, {int(vehicle.route_info['route_length']/1000)} км.")

                # Преобразование геоданных маршрута в GeoJSON
                if vehicle.route_info['route_gdf'] is not None:
                    route_geojson = vehicle.route_info['route_gdf'].to_json()

                    # Добавление маршрута на карту
                    folium.GeoJson(route_geojson, style_function=lambda x, color=route_color: {'color': color}).add_to(route_layer)

                # Группировка заказов по их географическому расположению
                location_to_orders = defaultdict(list)
                for order in vehicle.orders[:-1]:
                    location = (order.latitude, order.longitude)
                    location_to_orders[location].append(order)

                # Добавление маркеров для каждой уникальной геолокации
                for location, orders in location_to_orders.items():
                    popup_text = "<br>".join([f'Заказ: {order.id}, Позиция: {order.delivery_sequence}, кластер: {order.cluster}' for order in orders])
                    folium.Marker(
                        location=location,
                        popup=popup_text,
                        icon=folium.Icon(color=route_color, icon='info-sign')
                    ).add_to(route_layer)

                route_layer.add_to(m)

        # Добавление контроля слоев на карту
        folium.LayerControl().add_to(m)
        return m


class AntColonyOptimizer:
    def __init__(self, vehicle, matrix, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.vehicle = vehicle
        self.original_matrix = matrix
        self.node_to_index, self.distance_matrix = self._process_matrix()
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

   
    def _process_matrix(self):
        unique_nodes = list(self.original_matrix.keys())
        node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
        n = len(unique_nodes)
        distance_matrix = np.full((n, n), float('inf'))

        for node1 in unique_nodes:
            for node2 in unique_nodes:
                i, j = node_to_index[node1], node_to_index[node2]
                distance_matrix[i][j] = self.original_matrix[node1][node2]

        return node_to_index, distance_matrix

    def run(self):
        if self.vehicle.orders:  # Проверка на наличие заказов у автомобиля
            # Преобразование номеров узлов в индексы и создание подматрицы расстояний
            order_indices = [self.node_to_index[order.node] for order in self.vehicle.orders]
            self.distances = self.distance_matrix[np.ix_(order_indices, order_indices)]

            # Сохранение отображения индексов обратно в номера узлов
            index_to_real_node = {i: order.node for i, order in enumerate(self.vehicle.orders)}

            # Инициализация феромонов и параметров
            self.pheromone = np.ones(self.distances.shape) / len(self.distances)
            self.all_inds = range(len(self.distances))

            best_path_length = np.inf
            for _ in range(self.n_iterations):
                all_paths = self.gen_all_paths()
                self.spread_pheromone(all_paths)
                current_best_path, current_best_path_length = min(all_paths, key=lambda x: x[1])
                if current_best_path_length < best_path_length:
                    best_path_length = current_best_path_length
                    best_path = current_best_path
                self.pheromone *= self.decay

            # Преобразование индексов лучшего маршрута обратно в номера узлов
            best_path_nodes = [index_to_real_node[idx] for idx in best_path]
            return best_path_nodes, best_path_length
        else:
            st.write(f"Vehicle {self.vehicle.id} has no orders.")
            return [], 0

    def spread_pheromone(self, all_paths):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:self.n_best]:
            for i in range(1, len(path) - 1):  
                self.pheromone[path[i-1]][path[i]] += 1.0 / dist

    
    def gen_path_dist(self, path):
        total_dist = 0
        for i in range(1, len(path)):
            total_dist += self.distances[path[i-1]][path[i]]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gen_path()
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self):
        path_pairs = []
        visited = set([0, len(self.distances) - 1])  # Исключить первую и последнюю точки из посещения
        prev = 0  # Начать с первой точки

        for _ in range(1, len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path_pairs.append((prev, move))
            prev = move
            visited.add(move)

        path_pairs.append((prev, len(self.distances) - 1))  # Добавить возвращение в последнюю точку

        # Преобразование пар узлов в список узлов без повторений
        path = [path_pairs[0][0]] + [pair[1] for pair in path_pairs]
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0  # Замена феромона в посещенных узлах на 0

        # Защита от деления на ноль
        dist[dist == 0] = 1e-10

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)
        
        # Замена NaN на маленькие числа, если они есть
        row = np.nan_to_num(row, nan=1e-10)

        norm_row = row / row.sum()
        move = np.random.choice(self.all_inds, 1, p=norm_row)[0]
        return move
    


def detect_encoding_separator_decimal(uploaded_file):
    # Читаем содержимое файла для определения кодировки
    content = uploaded_file.read(10000)
    result = chardet.detect(content)
    encoding = result['encoding']
    uploaded_file.seek(0)  # Возвращаем указатель в начало файла

    # Читаем содержимое файла снова, на этот раз как текст, для определения разделителя
    sample = uploaded_file.read(10000).decode(encoding)
    uploaded_file.seek(0)  # Возвращаем указатель в начало файла
    possible_separators = [',', ';', '\t', '|']
    separator_counts = {sep: sample.count(sep) for sep in possible_separators}
    separator = max(separator_counts, key=separator_counts.get, default=',')
    
    # Определение десятичного разделителя
    decimal = ',' if separator == ';' else '.'

    return encoding, separator, decimal

def try_read_csv(uploaded_file):
    encoding, separator, decimal = detect_encoding_separator_decimal(uploaded_file)
    # Преобразуем UploadedFile в StringIO для последующего чтения pandas
    file_buffer = io.StringIO(uploaded_file.getvalue().decode(encoding))
    try:
        df = pd.read_csv(file_buffer, sep=separator, decimal=decimal)
        return df, None
    except Exception as e:
        return None, e

def main():
    header_container = st.container()
    with header_container:
        st.markdown("### Оптимизация Маршрутов Доставки")

    # Использование контейнера для остального содержимого
    with st.sidebar:
        st.title("Параметры оптимизации")

         # Загрузка файлов
        uploaded_car_file = st.sidebar.file_uploader("Загрузите файл car.csv", type=['csv'])
        uploaded_delivery_file = st.sidebar.file_uploader("Загрузите файл delivery.csv", type=['csv'])


        RADIUS = st.number_input("Радиус домашней области (км)", min_value=1, value=20, step=1)
        num_clusters_outside_EKT_MIN = st.number_input("Мин. количество кластеров за пределами домашней области", min_value=1, value=5, step=1)
        num_clusters_outside_EKT_MAX = st.number_input("Макс. количество кластеров за пределами домашней области", min_value=1, value=5, step=1)
        max_delivery_points_per_route_outside = st.number_input("Макс. количество точек доставки для одного маршрута за пределами домашней области", min_value=1, value=100, step=1)
        num_clusters_EKT_MIN = st.number_input("Мин. количество кластеров в домашней области", min_value=1, value=3, step=1)
        num_clusters_EKT_MAX = st.number_input("Макс. количество кластеров в домашней области", min_value=1, value=3, step=1)
        max_delivery_points_per_route_city = st.number_input("Макс. количество точек доставки для одного маршрута в домашней области", min_value=1, value=15, step=1)

        # Кнопка отправки формы
        submitted = st.button("Запустить оптимизацию")

        if submitted:    
            with header_container:
                if uploaded_car_file and uploaded_delivery_file:
                    df_trucks, error = try_read_csv(uploaded_car_file)
                    if df_trucks is not None:
                        truck_names = ['truck', 'truck_weight', 'truck_volume', 'truck_palletes']
                        df_trucks.columns = truck_names[:len(df_trucks.columns)]  # Обновление в соответствии с предложенной структурой
                        st.success(f"Файл '{uploaded_car_file.name}' успешно обработан.")
                    else:
                        st.error(f"Ошибка при чтении файла '{uploaded_car_file.name}': {str(error)}")

                    df_route, error = try_read_csv(uploaded_delivery_file)
                    if df_route is not None:
                        ride_names = ['document', 'doc_weight', 'doc_vol', 'doc_palletes', 'lon', 'lat']
                        df_route.columns = ride_names[:len(df_route.columns)]  # Обновление в соответствии с предложенной структурой
                        df_route = df_route.dropna(subset=['lon', 'lat'])
                        st.success(f"Файл '{uploaded_delivery_file.name}' успешно обработан.")
                    else:
                        st.error(f"Ошибка при чтении файла '{uploaded_delivery_file.name}': {str(error)}")
                else:
                    if not uploaded_car_file:
                        st.error("Файл 'car.csv' не найден.")
                    if not uploaded_delivery_file:
                        st.error("Файл 'delivery.csv' не найден.")


                Points = DeliveryPointsManager() #Создание объекта класса DeliveryPointsManager
                Points.add_from_dataframe(df_route) # Загрузка данных  о точках доставки(заказах) из датафрейма
                City = EkaterinburgDeliveryProcessor(Points)
                city_G = City.load_graph(RADIUS)
                City.assign_points_to_graph()
                st.write("Расчет матрицы расстояний домашнего региона")
                City.calculate_distance_matrix()

                # Основной итерационный процесс кластеризации заказов, их загрузки в авто, выбранные в случайном порядке и построение для каждой итерации оптимального маршрута муравьинным алгоритмом
                best_total_route_length = float('inf')
                best_parameters = None
                st.write("Запуск основного процесса обработки кластеров вне города, загрузки автомашин и поиска оптимальных маршрутов")
                for num_clusters_outside_EKT in range(num_clusters_outside_EKT_MIN, num_clusters_outside_EKT_MAX+1):
                    st.write(f" Обработка заказов за пределами домашнего региона. Обрабатываемое количество кластеров точек доставки: {num_clusters_outside_EKT}")
                    Outside = OutsideCityDeliveryProcessor(Points)
                    Outside.cluster_outside_delivery_points(num_clusters_outside_EKT)
                    st.write(" Загрузка дорожных графов для каждого кластера")
                    Outside.load_cluster_graphs()
                    Outside.assign_points_to_graph()
                    st.write(" Расчет матрицы расстояний для точек доставки каждого кластера на основе дорожных графов")
                    Outside.calculate_distance_matrices()

                    for num_clusters_EKT in range(num_clusters_EKT_MIN, num_clusters_EKT_MAX+1):
                        st.write(f"  Обработка заказов в домашнем регионе. Обрабатываемое количество кластеров точек доставки: {num_clusters_EKT}")
                        City.cluster_delivery_points(num_clusters_EKT, num_clusters_outside_EKT)

                        for iteration in range(1, 51):
                            #print("    Итерация - ", iteration)
                            Trucks = VehiclesManager()
                            Trucks.add_from_dataframe(df_trucks)
                            Trucks.load_vehicles_with_orders(Points, randomize=True, 
                                                            max_delivery_points_per_route_city=max_delivery_points_per_route_city, 
                                                            max_delivery_points_per_route_outside=max_delivery_points_per_route_outside, 
                                                            log_file=False)

                            Trucks.add_warehouse_to_route(city_G)
                            Trucks.adjust_warehouse_order_for_outside_routes(Outside)

                            all_route_length = 0

                            for vehicle in Trucks.vehicles:
                                if vehicle.orders:
                                    if all(order.in_ekaterinburg == False for order in vehicle.orders):
                                        cluster_label = vehicle.orders[0].cluster
                                        matrix = Outside.distance_matrices[cluster_label]
                                    elif all(order.in_ekaterinburg == True for order in vehicle.orders):
                                        matrix = City.distance_matrix

                                    aco = AntColonyOptimizer(vehicle, matrix, n_ants=10, n_best=5, n_iterations=10, decay=0.5, alpha=1, beta=5)
                                    best_path_nodes, best_path_length = aco.run()
                                    best_path_length += vehicle.closest_order_distance * 2
                                    vehicle.route_info['route_nodes'] = best_path_nodes
                                    all_route_length += best_path_length

                            if all_route_length < best_total_route_length:
                                best_total_route_length = all_route_length
                                # Создаем "глубокие" копии объектов
                                best_trucks = copy.deepcopy(Trucks)
                                best_outside = copy.deepcopy(Outside)
                                best_city = copy.deepcopy(City)
                                best_points = copy.deepcopy(Points)
                                
                                # Сохраняем копии объектов в best_parameters
                                best_parameters = (num_clusters_outside_EKT, num_clusters_EKT, iteration, best_trucks, best_outside, best_city, best_points)

                                
                st.success(f"Лучший суммарный маршрут: {best_total_route_length} метров")
                st.success(f"Лучшие параметры: Кластеры вне города - {best_parameters[0]}, Кластеры в городе - {best_parameters[1]}, Итерация - {best_parameters[2]}")

                if best_parameters:
                    # Восстановление состояния объектов из лучшей итерации
                    best_trucks, best_outside, best_city, best_points = best_parameters[3], best_parameters[4], best_parameters[5], best_parameters[6]
                    Trucks = best_trucks
                    Outside = best_outside
                    City = best_city
                    Points = best_points

                for vehicle in Trucks.vehicles:
                    # Определяем, используем ли мы городской граф или граф кластера
                    if all(order.in_ekaterinburg for order in vehicle.orders):
                        G = City.G  # Городской граф
                        matrix = City.distance_matrix
                    else:
                        cluster_label = vehicle.orders[0].cluster
                        G = Outside.cluster_graphs.get(cluster_label)  # Граф кластера
                        matrix = Outside.distance_matrices.get(cluster_label)
                    #print(vehicle.id)
                    if G and matrix and 'route_nodes' in vehicle.route_info and vehicle.route_info['route_nodes']:

                        if all(order.in_ekaterinburg for order in vehicle.orders):
                            # Городская логика обработки заказов
                            assigned_delivery_numbers = {}
                            delivery_number = 0
                            for node in vehicle.route_info['route_nodes']:
                                if node not in assigned_delivery_numbers:
                                    assigned_delivery_numbers[node] = delivery_number
                                    delivery_number += 1

                            for order in vehicle.orders:
                                if order.node in assigned_delivery_numbers:
                                    order.delivery_sequence = assigned_delivery_numbers[order.node]
                                    
                        else:
                            # Внегородская логика обработки заказов
                            assigned_delivery_numbers = {}
                            delivery_number = 1
                            for node in vehicle.route_info['route_nodes']:
                                if node not in assigned_delivery_numbers:
                                    assigned_delivery_numbers[node] = delivery_number
                                    delivery_number += 1

                            for order in vehicle.orders:
                                if order.node in assigned_delivery_numbers:
                                    order.delivery_sequence = assigned_delivery_numbers[order.node]

                        # Построение итогового маршрута
                        route_nodes = vehicle.route_info['route_nodes']
                        all_nodes = [route_nodes[0]]
                        vehicle.route_info = {
                            'route': [],
                            'route_gdf': None,
                            'route_length': 0,
                            'route_sequence': []  
                        }

                        for i in range(len(route_nodes) - 1):
                            node = route_nodes[i]
                            next_node = route_nodes[i + 1]

                            # Находим кратчайший путь между узлами
                            path = nx.shortest_path(G, node, next_node, weight='length')
                            all_nodes.extend(path[1:])

                            # Вычисляем длину пути
                            length = matrix[node][next_node]
                            vehicle.route_info['route_length'] += length

                            # Сохраняем информацию о пути
                            vehicle.route_info['route_sequence'].append({
                                'from': node,
                                'to': next_node,
                                'path_nodes': path,
                                'distance': length
                            })
                        vehicle.route_info['route_length'] += vehicle.closest_order_distance*2
                        # Преобразуем маршрут в GeoDataFrame
                        if len(all_nodes) > 1:
                            route_subgraph = G.subgraph(all_nodes)
                            _, edges_gdf = ox.utils_graph.graph_to_gdfs(route_subgraph)
                            vehicle.route_info['route_gdf'] = edges_gdf
                        else:
                            # Обработка случая с одним узлом (например, маршрут не был построен)
                            #print(f"В маршруте автомобиля {vehicle.id} всего один узел. GeoDataFrame не создан.")
                            vehicle.route_info['route_gdf'] = None

                import pandas as pd
                import io

                def create_delivery_points_csv():
                    columns = [
                        'УИД_АВТОМОБИЛЯ', 'ВЕС', 'ОБЪЕМ', 'КОЛИЧЕСТВО_ПАЛЕТ', 'УИД_ДОКУМЕНТА',
                        'ВЕС_ДОКУМЕНТА', 'ОБЪЁМ_ДОКУМЕНТА', 'КОЛ_ВО_ПАЛЕТ_ДОКУМЕНТА',
                        'ДОЛГОТА', 'ШИРОТА', 'ПОРЯДОК_РАЗВОЗА'
                    ]

                    data = []

                    for vehicle in Trucks.vehicles:
                        if vehicle.orders:
                            # Для автомобилей вне города исключаем последний заказ
                            if all(not order.in_ekaterinburg for order in vehicle.orders):
                                sorted_orders = sorted(vehicle.orders[:-1], key=lambda o: o.delivery_sequence)
                            else:
                                sorted_orders = sorted(vehicle.orders, key=lambda o: o.delivery_sequence)

                            for order in sorted_orders:
                                if not isinstance(order, LoadingPoint):
                                    for individual_order in order.individual_orders:
                                        row = [
                                            vehicle.id,
                                            vehicle.capacity,
                                            vehicle.volume,
                                            vehicle.max_pallets,
                                            individual_order['id'],
                                            individual_order['weight'],
                                            individual_order['volume'],
                                            individual_order['pallets'],
                                            order.longitude,
                                            order.latitude,
                                            order.delivery_sequence
                                        ]
                                        data.append(row)

                    df = pd.DataFrame(data, columns=columns)

                    # Создаем строку "Суммарная длина маршрутов" и добавляем None для всех остальных столбцов
                    #total_route_length_row = ["Суммарная длина маршрутов (метры)"] + [None] * (len(columns) - 2)

                    # Добавляем значение переменной best_total_route_length
                    #total_route_length_row.append(best_total_route_length)

                    # Добавляем строку "Суммарная длина маршрутов" и значение из переменной best_total_route_length
                    #df.loc[len(df)] = total_route_length_row

                    # Сохраняем DataFrame в CSV формате в буфер
                    output = io.StringIO()
                    df.to_csv(output, index=False, encoding='utf-8-sig', sep=';')
                    output.seek(0)
                    return output.getvalue()


                def plot_graph_boundaries_with_points(delivery_points_manager, radius_km):
                    """
                    Рисует прямоугольник, показывающий границы графа вокруг Екатеринбурга, и точки доставки по кластерам.
                    """
                    ekaterinburg_center = (56.838011, 60.597465)
                    map_center = ekaterinburg_center

                    # Создание карты
                    m = folium.Map(location=map_center, zoom_start=10)

                    # Расчет границ прямоугольника
                    delta_lat = radius_km / 111
                    delta_lon = radius_km / (111 * cos(ekaterinburg_center[0] * pi / 180))

                    # Координаты углов прямоугольника
                    bounds = [
                        (ekaterinburg_center[0] - delta_lat, ekaterinburg_center[1] - delta_lon),
                        (ekaterinburg_center[0] + delta_lat, ekaterinburg_center[1] + delta_lon)
                    ]

                    # Рисуем прямоугольник на карте
                    folium.Rectangle(bounds, color="green", fill=True, fill_opacity=0.1).add_to(m)

                    # Цвета для разных кластеров
                    colors = cycle(['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'black', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray'])

                    # Создание словаря для соответствия кластеров и цветов
                    cluster_colors = {}
                    for point in delivery_points_manager.delivery_points:
                        if point.cluster not in cluster_colors:
                            cluster_colors[point.cluster] = next(colors)

                    # Добавление маркеров точек доставки
                    for point in delivery_points_manager.delivery_points:
                        point_color = cluster_colors[point.cluster]
                        popup_text = f'Заказ: {point.id}, Кластер: {point.cluster}'
                        folium.Marker(
                            location=(point.latitude, point.longitude),
                            popup=popup_text,
                            icon=folium.Icon(color=point_color, icon='info-sign')
                        ).add_to(m)

                    return m
                
                def create_map_html_file(map_object):
                    """ Сохраняет карту в объект BytesIO в формате HTML. """
                    with tempfile.NamedTemporaryFile() as tmp_file:
                        # Сохраняем карту во временный файл
                        map_object.save(tmp_file.name)

                        # Считываем содержимое временного файла в BytesIO
                        tmp_file.seek(0)
                        html_file = io.BytesIO(tmp_file.read())

                    return html_file
                             
                csv_file = create_delivery_points_csv()
                m1 = plot_graph_boundaries_with_points(Points, RADIUS)
                st.write("Карта кластеризации заказов и граница домашнего региона:")
                folium_static(m1, width=1000, height=700)
                map_html1 = create_map_html_file(m1)
                m2 = Trucks.plot_routes_on_map()
                map_html2 = create_map_html_file(m2)
                st.write("Карта  маршрутов заказов:")
                folium_static(m2, width=1000, height=700)

                # Создание ZIP-архива для скачивания
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
                    # Добавление файла Excel
                    zip_file.writestr('Загрузка машин и порядок доставки.csv', csv_file)

                    # Добавление карт в формате HTML
                    zip_file.writestr('Кластеры и домашний регион.html', map_html1.getvalue())
                    zip_file.writestr('Карта маршрутов.html', map_html2.getvalue())

                zip_buffer.seek(0)
                st.download_button('Скачать файлы: Загрузка машин и порядок доставки.csv, Кластеры и домашний регион.html и Карта маршрутов.html одним файлом (ZIP). Страница будет перезагружена и очищена.', zip_buffer, 'all_files.zip', 'application/zip')

if __name__ == "__main__":
    main()    