# Link: https://github.com/udacity/deep-reinforcement-learning/tree/master/tile-coding

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


plt.style.use('ggplot')
np.set_printoptions(precision=3, linewidth=120)


def create_tiling_grid(low, high, bins=(10, 10), offsets=(0.0, 0.0)):
  """ Define um grid uniformement espaçado que pode ser usado para tile-coding """

  grid = [np.linspace(low[dim], high[dim], bins[dim] + 1)[1:-1] + offsets[dim] for dim in range(len(bins))]
  print("Tiling: [<low>, <high>] / <bins> + (<offset>) => <splits>")
  for l, h, b, o, splits in zip(low, high, bins, offsets, grid):
    print("    [{}, {}] / {} + ({}) => {}".format(l, h, b, o, splits))
  return grid


def create_tilings(low, high, tiling_specs):
  """ Define os tilings usando as especificações """
  return [create_tiling_grid(low, high, bins, offsets) for bins, offsets in tiling_specs]


def visualize_tilings(tilings):
    """Plot each tiling as a grid."""
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    linestyles = ['-', '--', ':']
    legend_lines = []

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, grid in enumerate(tilings):
        for x in grid[0]:
            l = ax.axvline(x=x, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], label=i)
        for y in grid[1]:
            l = ax.axhline(y=y, color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)])
        legend_lines.append(l)
    ax.grid('off')
    ax.legend(legend_lines, ["Tiling #{}".format(t) for t in range(len(legend_lines))], facecolor='white', framealpha=0.9)
    ax.set_title("Tilings")
    return ax


def discretize(sample, grid):
  """ Discretiza uma amostra """
  return tuple(int(np.digitize(s, g)) for s, g in zip(sample, grid))


def tile_encode(sample, tilings, flatten=False):
  """ Codifica uma amostra usando o tile-coding """
  encoded_sample = [discretize(sample, grid) for grid in tilings]
  return np.concatenate(encoded_sample) if flatten else encoded_sample


def visualize_encoded_samples(samples, encoded_samples, tilings):
  """ Visualizar amostras ativando as tiles respectivas """
  samples = np.array(samples)

  ax = visualize_tilings(tilings)
  ax.plot(samples[:, 0], samples[:, 1], 'o', alpha=0.0)
  low = [ax.get_xlim()[0], ax.get_ylim()[0]]
  high = [ax.get_xlim()[1], ax.get_ylim()[1]]

  tilings_extended = [np.hstack((np.array([low]).T, grid, np.array([high]).T)) for grid in tilings]
  tile_centers = [(grid_extended[:, 1:] + grid_extended[:, :-1]) / 2 for grid_extended in tilings_extended]
  tile_toplefts = [grid_extended[:, :-1] for grid_extended in tilings_extended]
  tile_bottomrights = [grid_extended[:, 1:] for grid_extended in tilings_extended]

  ax.plot(samples[:, 0], samples[:, 1], 'o', color='r')
  prop_cycle = plt.rcParams['axes.prop_cycle']
  colors = prop_cycle.by_key()['color']
  for sample, encoded_sample in zip(samples, encoded_samples):
    for i, tile in enumerate(encoded_sample):
      topleft = tile_toplefts[i][0][tile[0]], tile_toplefts[i][1][tile[1]]
      bottomright = tile_bottomrights[i][0][tile[0]], tile_bottomrights[i][1][tile[1]]
      ax.add_patch(Rectangle(topleft, bottomright[0] - topleft[0], bottomright[1] - topleft[1],
                             color=colors[i], alpha=0.33))

      if any(sample < topleft) or any(sample > bottomright):
        cx, cy = tile_centers[i][0][tile[0]], tile_centers[i][1][tile[1]]
        ax.add_line(Line2D([sample[0], cx], [sample[1], cy], color=colors[i]))
        ax.plot(cx, cy, 's', color=colors[i])

  ax.margins(x=0, y=0)
  ax.set_title("Amostras com a tile-encoding")
  return ax


if __name__ == '__main__':
  low = [-1.0, -5.0]
  high = [1.0, 5.0]

  # Tiling specs: [(<bins>, <offsets>), ...]
  tiling_specs = [((10, 10), (-0.066, -0.33)),
                  ((10, 10), (0.0, 0.0)),
                  ((10, 10), (0.066, 0.33))]
  tilings = create_tilings(low, high, tiling_specs)

  visualize_tilings(tilings)
  plt.show()

  # Testa a codificação de algumas amostras
  samples = [(-1.2, -5.1),
             (-0.75, 3.25),
             (-0.5, 0.0),
             (0.25, -1.9),
             (0.15, -1.75),
             (0.75, 2.5),
             (0.7, -3.7),
             (1.0, 5.0)]
  encoded_samples = [tile_encode(sample, tilings) for sample in samples]
  print("\nAmostras:", repr(samples), sep="\n")
  print("\nAmostras codificadas:", repr(encoded_samples), sep="\n")

  visualize_encoded_samples(samples, encoded_samples, tilings)
  plt.show()