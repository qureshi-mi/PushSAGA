########################################################################################################################
####---------------------------------------------------Utilities----------------------------------------------------####
########################################################################################################################

## Used to pre-set networkx-class properties

def monitor(name,current,total):
    if (current+1) % (total/10) == 0:
        print ( name + ' %d%% completed' % int(100*(current+1)/total) )

def nx_options():
    options = {
     'node_color': 'skyblue',
     'node_size': 10,
     'edge_color': 'grey',
     'width': 0.5,
     'arrows': False,
     'node_shape': 'o',}
    return options