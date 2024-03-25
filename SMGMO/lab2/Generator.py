import numpy as np
import math


def generate(points_count, noise, type):
    if type == "circle":
        return circle_generation(points_count, noise)
    elif type == "gauss":
        return gauss_generation(points_count, noise)


def circle_generation(points_count, noise):
    radius = 1.0
    inner_radius = 1.5 * radius
    outer_radius = 2.0 * radius
    first_set = generate_points_in_circle(radius, points_count, noise, 1)
    second_set = generate_points_in_ring(inner_radius, outer_radius, points_count, noise, 0)
    merged_result = np.vstack((first_set, second_set))
    np.random.shuffle(merged_result)
    return merged_result


def generate_points_in_circle(rad, points_count, noise, class_value):
    r = rad * np.sqrt(np.random.rand(points_count))
    theta = np.random.uniform(0, 2 * math.pi, points_count)
    x = r * np.cos(theta) + np.random.uniform(-noise, noise, points_count)
    y = r * np.sin(theta) + np.random.uniform(-noise, noise, points_count)
    c = np.full(points_count, class_value)
    return np.column_stack((x, y, c))


def generate_points_in_ring(in_radius, out_radius, points_count, noise, class_value):
    r = (out_radius - in_radius) * np.sqrt(np.random.rand(points_count)) + in_radius
    theta = np.random.uniform(0, 2 * math.pi, points_count)
    x = r * np.cos(theta) + np.random.uniform(-noise, noise, points_count)
    y = r * np.sin(theta) + np.random.uniform(-noise, noise, points_count)
    c = np.full(points_count, class_value)
    return np.column_stack((x, y, c))


def gauss_generation(points_count, noise):
    center_x1, center_y1 = 3, 3
    center_x2, center_y2 = -3, -3
    first_set = generate_gauss_set(center_x1, center_y1, points_count, noise, 1)
    second_set = generate_gauss_set(center_x2, center_y2, points_count, noise, 0)
    merged_result = np.vstack((first_set, second_set))
    np.random.shuffle(merged_result)
    return merged_result


def generate_gauss_set(center_x, center_y, points_count, noise, class_value):
    x = np.random.normal(center_x, 1, points_count)
    y = np.random.normal(center_y, 1, points_count)
    x += np.random.uniform(-noise, noise, points_count)
    y += np.random.uniform(-noise, noise, points_count)
    c = np.full(points_count, class_value)
    return np.column_stack((x, y, c))
