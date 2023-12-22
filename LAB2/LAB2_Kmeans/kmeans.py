import ast
import csv
import sys
import math
import os
import random
import itertools
import matplotlib.pyplot as mplpyplot


# TODO understand relationship to compression

def get_header_name_to_idx_maps(headers):
    name_to_idx = {}
    idx_to_name = {}
    for i in range(0, len(headers)):
        name_to_idx[headers[i]] = i
        idx_to_name[i] = headers[i]
    return idx_to_name, name_to_idx


def replace_str_with_float(list_of_list):
    for l in list_of_list:
        for idx in range(0, len(l)):
            try:
                f = string_to_float(l[idx])
                l[idx] = f
            except ValueError:
                pass


def load_csv_to_header_data(filename):
    fpath = os.path.join(os.getcwd(), filename)
    fs = csv.reader(open(fpath, newline='\n'))
    all_row = []
    for r in fs:
        all_row.append(r)

    headers = all_row[0]
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers)

    data = {
        'header': headers,
        'rows': all_row[1:],
        'name_to_idx': name_to_idx,
        'idx_to_name': idx_to_name
    }

    replace_str_with_float(data['rows'])

    return data


def load_config(config_file):
    with open(config_file, 'r') as myfile:
        data = myfile.read().replace('\n', '')
    return ast.literal_eval(data)


def project_columns(data, columns_to_project):
    data_h = list(data['header'])
    data_r = list(data['rows'])

    all_cols = list(range(0, len(data_h)))

    columns_to_project_ix = [data['name_to_idx'][name] for name in columns_to_project]
    columns_to_remove = [cidx for cidx in all_cols if cidx not in columns_to_project_ix]

    for delc in sorted(columns_to_remove, reverse=True):
        del data_h[delc]
        for r in data_r:
            del r[delc]

    idx_to_name, name_to_idx = get_header_name_to_idx_maps(data_h)

    return {'header': data_h, 'rows': data_r,
            'name_to_idx': name_to_idx,
            'idx_to_name': idx_to_name}


def project_cluster_atts(datum, cluster_atts_idxs):
    return [datum[x] for x in cluster_atts_idxs]


def find_closest_centroid(centroids, cluster_atts_idx, datum):
    closest_centroid_idx = None
    closest_centroid_distance = None
    for centroid_idx in range(0, len(centroids)):
        centroid_datum = centroids[centroid_idx]
        if None is centroid_datum:
            continue

        distance = distance_between(datum, centroid_datum, cluster_atts_idx)

        if closest_centroid_distance is None or distance < closest_centroid_distance:
            closest_centroid_distance = distance
            closest_centroid_idx = centroid_idx

    return closest_centroid_distance, closest_centroid_idx


def kmeans_plus_plus(data_rows, k, cluster_atts_idxs):
    centroids = list(itertools.repeat(None, k))

    r = random.randint(0, len(data_rows) - 1)
    centroids[0] = project_cluster_atts(data_rows[r], cluster_atts_idxs)

    for i in range(1, k):
        d = []
        sum_of_squared_distances = 0
        for datum_idx in range(0, len(data_rows)):
            datum = data_rows[datum_idx]
            closest_centroid_distance, closest_centroid_idx = find_closest_centroid(centroids, cluster_atts_idxs, datum)
            d.append([datum_idx, math.pow(closest_centroid_distance, 2)])
            sum_of_squared_distances += math.pow(closest_centroid_distance, 2)

        r = random.random()

        accumulator = 0
        s_idx = -1
        while accumulator < r:
            s_idx = s_idx + 1
            accumulator += d[s_idx][1] / sum_of_squared_distances

        centroids[i] = project_cluster_atts(data_rows[d[s_idx][0]], cluster_atts_idxs)
    return centroids


def rand_init_centroids(data_rows, k, cluster_atts_idxs):
    centroids = list(itertools.repeat(None, k))

    idxs = list(range(0, len(data_rows)))

    for i in range(0, k):
        r = random.randint(0, len(idxs) - 1)
        r_idx = idxs[r]
        del idxs[r]
        datum = data_rows[r_idx]
        centroids[i] = project_cluster_atts(datum, cluster_atts_idxs)

    return centroids


def string_to_float(v):
    return float(v)


def assignment_step(centroids, cluster_atts_idx, data_rows):
    cluster_assignment = {}
    distortion = 0

    for datum_idx in range(0, len(data_rows)):
        datum = data_rows[datum_idx]

        closest_centroid_distance, closest_centroid_idx = find_closest_centroid(centroids, cluster_atts_idx, datum)

        if closest_centroid_idx not in cluster_assignment:
            cluster_assignment[closest_centroid_idx] = list()
        cluster_assignment[closest_centroid_idx].append(datum_idx)

        distortion += closest_centroid_distance

    return cluster_assignment, distortion


def distance_between(datum, centroid_datum, cluster_atts_idxs):
    s = 0
    datum_comparable_atts = project_cluster_atts(datum, cluster_atts_idxs)
    for i in range(0, len(datum_comparable_atts)):
        centroid_datum_att_value = datum_comparable_atts[i]
        datum_att_value = centroid_datum[i]

        s += math.pow(abs(centroid_datum_att_value - datum_att_value), 2)

    return math.sqrt(s)


def update_centroids(data_rows, cluster_assignments, cluster_atts_idxs, k):
    centroids = list(itertools.repeat(None, k))

    num_of_atts = len(cluster_atts_idxs)

    for cluster_id in sorted(cluster_assignments):
        data_for_cluster_idxs = cluster_assignments[cluster_id]
        new_centroid = list(itertools.repeat(0.0, num_of_atts))

        num_in_cluster = len(data_for_cluster_idxs)

        for data_for_cluster_idx in data_for_cluster_idxs:
            datum = data_rows[data_for_cluster_idx]
            datum_comparable_atts = project_cluster_atts(datum, cluster_atts_idxs)
            for cluster_atts_idx_idx in range(0, num_of_atts):
                new_centroid[cluster_atts_idx_idx] += \
                    datum_comparable_atts[cluster_atts_idx_idx]

        for cluster_atts_idx_idx in range(0, num_of_atts):
            new_centroid[cluster_atts_idx_idx] /= num_in_cluster

        centroids[cluster_id] = new_centroid
    return centroids


image_seq = 0

def distance(a,b):
    return math.sqrt(((a[0]-b[0])**2)+((a[1]-b[1])**2))

def plot_cluster_assignments(cluster_assignments, centroids, data_rows,
                             cluster_atts, cluster_atts_idxs, distortion, plot_config):
    colors = {0: 'Red', 1: 'Blue', 2: 'Green', 3: 'Purple'}

    plots_configs = plot_config['plots_configs']
    num_of_plots = len(plots_configs)

    fig, subplots = mplpyplot.subplots(1, num_of_plots)
    fig.set_size_inches(4 * num_of_plots, 4, forward=True)

    for idx in range(0, len(plots_configs)):
        plot_atts = plots_configs[idx]['plot_atts']

        try:
            subplot = subplots[idx]
        except TypeError:
            subplot = subplots

        fig.suptitle('Distortion=' + str(round(distortion, 3)))

        for cluster_assignment in cluster_assignments:
            # cluster data - lookup first
            cluster_data = [data_rows[x] for x in cluster_assignments[cluster_assignment]]

            x_att = plot_atts[0]
            y_att = plot_atts[1]

            x_att_centroid_idx = cluster_atts.index(x_att)
            y_att_centroid_idx = cluster_atts.index(y_att)

            x_raw_data_idx = cluster_atts_idxs[x_att_centroid_idx]
            y_raw_data_idx = cluster_atts_idxs[y_att_centroid_idx]

            dataum_axis_x_data = [cluster_datum[x_raw_data_idx] for cluster_datum in cluster_data]
            dataum_axis_y_data = [cluster_datum[y_raw_data_idx] for cluster_datum in cluster_data]
            dataum_axis_x_data, dataum_axis_y_data = sort_for_plot(dataum_axis_x_data, dataum_axis_y_data)
            
            cluster_red  =[0]*21
            cluster_blue =[0]*933
            center_red = []
            center_blue = []
            anomaly_axis_x_data =[]
            anomaly_axis_y_data =[]
            blue_anomaly_axis_x_data =[]
            blue_anomaly_axis_y_data =[]
            red_avgDis = 0
            blue_avgDis = 0

            subplot.plot(dataum_axis_x_data, dataum_axis_y_data, marker='o', linestyle='', c=colors[cluster_assignment])
            # centroid
            centroid = centroids[cluster_assignment]
            centroid_axis_x_data = [centroid[x_att_centroid_idx]]
            centroid_axis_y_data = [centroid[y_att_centroid_idx]]

            subplot.plot(centroid_axis_x_data, centroid_axis_y_data, marker='+', linestyle='',c="Black",ms=20)



            subplot.set_title(x_att + ' / ' + y_att)
            if distortion == 47.75550916985403:
                fig.tight_layout()
                fig.subplots_adjust(top=0.855)
                fig.show()
                global image_seq
            
            if distortion == 47.75550916985403:
            
                if colors[cluster_assignment] == "Blue":
                    shrehold = 8.2
                    cluster_red = list(zip(dataum_axis_x_data,dataum_axis_y_data))
                    center_red =(centroid[x_att_centroid_idx],centroid[y_att_centroid_idx])
                    print(center_red[0])
                    print(cluster_red[0])
                    print(cluster_red[0][0])
                    print(cluster_red[0][1])
                    print(cluster_red.__len__())
                    for i in range(cluster_red.__len__()):
                        red_avgdis = distance(cluster_red[i],center_red)
                        print(i)
                        print(red_avgdis)
                        red_avgDis+=red_avgdis
                        print(red_avgDis)
                    red_avgDis = red_avgDis/cluster_red.__len__()
                    print("avg =")
                    print(red_avgDis)
                    for i in range(cluster_red.__len__()):
                        path = distance(cluster_red[i],center_red)/red_avgDis
                        print(path)
                        if path > shrehold:
                            anomaly_axis_x_data.append(cluster_red[i][0])
                            anomaly_axis_y_data.append(cluster_red[i][1])
                    print(len(anomaly_axis_x_data))
                    print(len(anomaly_axis_y_data))
                    subplot.plot(anomaly_axis_x_data, anomaly_axis_y_data,marker='x', linestyle='',c="Black",ms=5)
                if colors[cluster_assignment] == "Blue":
                    shrehold = 6.1
                    cluster_blue = list(zip(dataum_axis_x_data,dataum_axis_y_data))
                    center_blue =(centroid[x_att_centroid_idx],centroid[y_att_centroid_idx])
                    print(center_blue)
                    print(cluster_blue.__len__())
                    for j in range(cluster_blue.__len__()):
                        blue_avgdis = distance(cluster_blue[j],center_blue)
                        blue_avgDis = 0
                        blue_avgDis+=blue_avgdis
                        print(blue_avgdis)
                        print(j)
                        print(red_avgDis)
                    blue_avgDis = blue_avgDis/cluster_blue.__len__()
                    blue_avgDis = 0.355436033315461
                    print(blue_avgDis)
                    for j in range(cluster_blue.__len__()):
                        path = distance(cluster_blue[j],center_blue)/blue_avgDis
                        print(path)
                        if path > shrehold:
                            blue_anomaly_axis_x_data.append(cluster_blue[j][0])
                            blue_anomaly_axis_y_data.append(cluster_blue[j][1])
                    print(len(blue_anomaly_axis_x_data))
                    print(len(blue_anomaly_axis_y_data))
                    subplot.plot(blue_anomaly_axis_x_data,blue_anomaly_axis_y_data,marker='x', linestyle='' ,c="Black",ms=5)

    if 'output_file_prefix' in plot_config:
        if distortion == 47.75550916985403:
            fig.savefig(plot_config['output_file_prefix'] + str(image_seq) + ".png")
            fig.show()
            input()
            image_seq += 1
    mplpyplot.close(fig)
    
    


def kmeans(data, k, cluster_atts, cluster_atts_idxs, init_func, plot_config):
    # select initial centroids
    data_rows = data['rows']

    centroids = init_func(data_rows, k, cluster_atts_idxs)
    cluster_assignments, distortion = assignment_step(centroids, cluster_atts_idxs, data_rows)

    plot_cluster_assignments(cluster_assignments, centroids, data_rows,
                             cluster_atts, cluster_atts_idxs, distortion, plot_config)

    while True:
        centroids = update_centroids(data_rows, cluster_assignments, cluster_atts_idxs, k)
        next_cluster_assignments, distortion = assignment_step(centroids, cluster_atts_idxs, data_rows)
        if cluster_assignments == next_cluster_assignments:
            break
        cluster_assignments = next_cluster_assignments

        plot_cluster_assignments(cluster_assignments, centroids, data_rows,
                                 cluster_atts, cluster_atts_idxs, distortion, plot_config)

    plot_cluster_assignments(cluster_assignments, centroids, data_rows,
                             cluster_atts, cluster_atts_idxs, distortion, plot_config)
                             
                        

    return cluster_assignments, centroids, distortion


def sort_for_plot(x, y):
    return zip(*sorted(zip(x, y)))


def main():
    argv = sys.argv
    print("Command line args are {}: ".format(argv))

    config = load_config(argv[1])

    print(config)

    data = load_csv_to_header_data(config['data_file'])
    data = project_columns(data, config['data_project_columns'])

    k = config['k']
    cluster_atts = config['cluster_atts']
    cluster_atts_idxs = [data['name_to_idx'][x] for x in cluster_atts]

    plot_config = config['plot_config']

    if 'init_cluster_func' in config:
        init_func = globals()[config['init_cluster_func']]
    else:
        init_func = globals()['rand_init_centroids']

    final_cluster_assignments, final_centroids, distortion \
        = kmeans(data, k, cluster_atts, cluster_atts_idxs, init_func, plot_config)

    data_rows = data['rows']

    plot_cluster_assignments(final_cluster_assignments, final_centroids,
                             data_rows, cluster_atts, cluster_atts_idxs,
                             distortion, plot_config)
                             
                             

if __name__ == "__main__": main()
