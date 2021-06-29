import numpy as np
import matplotlib.pyplot as plt


class Visualize:
    
    def __init__(self):
        pass
    
    def draw_deltas(self, deltas):

        """Рисует гистограммы - оценки распределений
            дельт (разностей метрик), по три в ряд.

            ------

            deltas - список массивов; каждый массив - это
                     набор дельт (разностей метрик)
                     в  фиксированном эксперименте,
                     повторенном определенное количество раз

            """

        n_graph_lines = len(deltas) // 3
        if len(deltas) % 3 != 0:
            n_graph_lines += 1

        fig, ax = plt.subplots(n_graph_lines, 3, sharey='row')
        fig.set_size_inches(12, 3 * n_graph_lines)
        fig.suptitle('Deltas', y=0, size=22)
        fig.subplots_adjust(hspace=0.4)

        if len(deltas) <= 3:
            ax = ax[np.newaxis, :]

        for i in range(len(deltas)):
            n_line = i // 3
            n_row = i % 3

            ax[n_line, n_row].hist(deltas[i], bins='rice', alpha=0.5,
                       histtype='stepfilled', density=True)
            ax[n_line, n_row].set_title(f'Delta for {i}-th metric', size=12);
        
        display(fig)

        return

