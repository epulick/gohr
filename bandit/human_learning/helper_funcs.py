import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from ast import literal_eval

def bucket_remap(by,bx):
    bucket = None

    if by==0:
        if bx==0:
            bucket=3
        else:
            bucket=2
    else:
        if bx==0:
            bucket=0
        else:
            bucket=1
    
    if bucket==None:
        print("Error mapping buckets")
        breakpoint()
    
    return bucket

def process_board(board_str):
    board = literal_eval(board_str)['value']
    smallest_id = board[0]['id']
    for piece in board:
        id = piece['id']
        if id<smallest_id:
            smallest_id=id
    board_copy=board
    return board_copy

def get_attributes(board, y, x):
    shape = None
    color = None
    id = None
    cell = None

    for piece in board:
        if piece['y']==y and piece['x']==x:
            shape = piece['shape']
            color = piece['color']
            id = piece['id']
            cell = (y-1)*6+x
            
    if shape==None or color==None or id==None or cell == None:
        print("Error evaluating attributes of pieces")
        breakpoint()

    # Zero indexing
    shapes = {'TRIANGLE':0,'CIRCLE':1,'SQUARE':2,'STAR':3}
    colors = {'RED':0,'BLUE':1,'YELLOW':2,'GREEN':3}
    return shape,color,shapes[shape], colors[color],id,cell,cell-1

def calc_availability(board,shape_order,color_order):
    shape_avail = np.full(4,np.nan)
    color_avail = np.full(4,np.nan)
    cell_avail = np.full(36,np.nan)

    for piece in board:
        color = piece['color']
        shape = piece['shape']
        y = piece['y']
        x = piece['x']
        cell = (y-1)*6+x
        shape_avail[shape_order.index(shape)]=shape_order.index(shape)
        color_avail[color_order.index(color)]=color_order.index(color)
        cell_avail[cell-1]=cell
    return np.concatenate((shape_avail,color_avail,cell_avail))

def ax_err_plots_player(ax,df):

    # Need to add a (0,0) entry to the data since it arrives beginning with the first move
    # Convert dataframe contents to numpy and prepend (0,0)
    x_vals = df.move.to_numpy()
    y_vals = df.cumulative_err.to_numpy()
    x_vals = np.insert(x_vals,0,0)
    y_vals = np.insert(y_vals,0,0)
    
    # Plot the data
    sns.lineplot(ax=ax, x=x_vals,y=y_vals,drawstyle='steps-post')
    
    # Plot lines for the 0,0 axis lines
    ax.axhline(0, color='gray',linewidth=0.25)
    ax.axvline(0, color='gray', linewidth=0.25)

    # Enforce that labels must be integers on the y-axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Axis/title formatting
    ax.set_xlabel('')
    ax.set_ylabel("Cumulative Error")
    return