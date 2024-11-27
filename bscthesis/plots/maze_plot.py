import os
import matplotlib.pyplot as plt

def generate_maze_plot(input_file, save_name, save_dir='images'):
    with open(input_file, 'r') as f:
        maze_lines = [line.rstrip('\n') for line in f]
    
    rows = len(maze_lines)
    cols = len(maze_lines[0]) if rows > 0 else 0
    wallmatrix = [[0 for _ in range(cols)] for _ in range(rows)]
    
    for i, row in enumerate(maze_lines):
        for j, c in enumerate(row):
            if c == ' ':
                wallmatrix[i][j] = 0
            else:
                wallmatrix[i][j] = 1
    
    fig, ax = plt.subplots(figsize=(cols, rows))
    for i in range(rows):
        for j in range(cols):
            if wallmatrix[i][j] == 1:
                x, y = j, rows - i
                if j < cols - 1 and wallmatrix[i][j + 1] == 1:
                    ax.plot([x, x + 1], [y, y], color='black', linewidth=2)
                if i < rows - 1 and wallmatrix[i + 1][j] == 1:
                    ax.plot([x, x], [y - 1, y], color='black', linewidth=2)

    ax.set_xlim(-0.03, cols -0.97)
    ax.set_ylim(0.97, rows + 0.03)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
