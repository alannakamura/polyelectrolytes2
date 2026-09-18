from polyelectrolytes import *
import numpy as np

p = Polieletrolito(gpu=False)

filename = '260902/fig11/'

inicio1 = time.perf_counter()

p.params['nx'] = int(2**12)
p.params['nx'] += 1
p.params['xn'] = 30*2
p.params['f'] = 0.14
p.params['v'] = 0.1
# p.params['c_salt'] = 3.01e-7
p.params['w2'] = 1500
p.params['phib2']=1e-6
p.params['relaxation_tax'] = 7e-3
p.params['error'] = 1e-4
p.params['dirichlet_boundary'] = [False, True, False, True]
# p.params['y0'] = -4*np.pi*p.params['lb']*1e-4*30
p.params['y0'] = -0.1
# p.params['yn'] = 0.0

t = ''
for i in p.params['dirichlet_boundary']:
    t = t+str(i).lower()[0]

filename += 'phib2_'
filename += str(p.params['phib2'])
filename += '_w2_'
filename += str(p.params['w2'])
filename += '_xn_'
filename += str(p.params['xn'])
filename += '_nx_'
filename += str(p.params['nx'])
filename += '_lambda_'
filename += str(p.params['relaxation_tax'])
filename += '_f_'
filename += str(p.params['f'])
filename += '_v_'
filename += str(p.params['v'])
filename += '_s_'
filename += str(p.params['c_salt'])
filename += '_y0_'
filename += str(p.params['y0'])
filename += '_t_'
filename += t
filename += '.pkl'

p.set_filename(filename)

p.calculate_constants()
print(p.params['k2'])
p.run()

fim = time.perf_counter()
tempo_total = fim - inicio1
print(f"tempo total: {tempo_total:.3f} s")