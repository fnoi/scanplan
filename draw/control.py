import plotly.graph_objects as go
import plotly.io
import numpy as np


def exec_plot(objlist, projection='orthographic', show=True, save=False, savefile=None):
    layout, dir_dict = update_layout(objlist=objlist)

    data = []
    for plot_object in objlist:
        data.extend(plot_object['data'])

    fig = go.Figure(data=data, layout=layout)
    fig.update_scenes(camera_projection_type=projection)

    fig.update_layout(
        scene_aspectmode='data',
        scene_aspectratio=dict(
            x=dir_dict['x']['aspect_ratio'],
            y=dir_dict['y']['aspect_ratio'],
            z=dir_dict['z']['aspect_ratio']
        ),
        margin=dict(
            t=5
        )
    )

    if show:
        fig.show()
    elif save:
        plotly.io.write_image(fig, savefile, format='png')
    else:
        raise 'what is the point if you do not show or save'


def update_layout(objlist):
    dirs = ['x', 'y', 'z']
    prec = 3
    fac_size = 1

    dir_dict = {}
    for direction in dirs:
        dir_dict[direction] = {}
        dir_dict[direction][f'min'] = round(min([np.min(_[direction]) for _ in objlist]), prec)
        dir_dict[direction][f'max'] = round(max([np.max(_[direction]) for _ in objlist]), prec)
        dir_dict[direction][f'range'] = [
            dir_dict[direction][f'min'],
            dir_dict[direction][f'max'],
            round(dir_dict[direction][f'max'] - dir_dict[direction][f'min'], prec),
            round(dir_dict[direction][f'min'] + 0.5 * abs(
                dir_dict[direction][f'max'] - dir_dict[direction][f'min']), prec
                  )
        ]
    axis_size = fac_size * max([dir_dict[_]['range'][2] for _ in dirs])
    for direction in dirs:
        dir_dict[direction][f'axis_range_{direction}'] = [
            dir_dict[direction]['range'][3] - axis_size/2,
            dir_dict[direction]['range'][3] + axis_size/2
        ]
        dir_dict[direction]['aspect_ratio'] = dir_dict[direction]['range'][2] / axis_size

    axis = dict(
        showbackground=False,
        showline=False,
        zeroline=False,
        showticklabels=False,
        title=''
    )
    layout = go.Layout(
        width=2000,
        height=1600,
        showlegend=False,
        scene=dict(xaxis=dict(axis),
                   yaxis=dict(axis),
                   zaxis=dict(axis),
                   ),
        margin=dict(
            t=100
        ),
        hovermode='closest'
    )

    return layout, dir_dict
