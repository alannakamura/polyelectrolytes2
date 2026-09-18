import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

diretorio = "260902/fig6_1/"
# diretorio = "200526/fig1/"
list_dir = os.listdir(diretorio)

atributo = ['phib2', 'w2', 'xn', 'nx', 'lambda', 'f', 'v','y0','s']
# atributo = ['phib2', 'w2', 'xn', 'nx', 'lambda', 'f', 'v']
atributo = atributo[-4]
atributo = list_dir[0].split('_').index(atributo)

# varios graficos de um diretorio
for i in range(len(list_dir)):
    list_dir[i] = list_dir[i]
    list_dir[i] = list((float(list_dir[i].strip('.pkl').split('_')[atributo+1]), list_dir[i]))
list_dir.sort(reverse=False)

# pegar uma faixa de graficos de um diretorio
# inicio = -2
# fim = -1
# list_dir = [list_dir[0]]
# list_dir = list_dir[-2:]

# simulacao de 1 grafico
# list_dir = 'teste_artigo1_phib2_4e-05_w2_0.0_xn_120_nx_4097_lambda_0.007_f_0.1.pkl'
# list_dir = 'teste_artigo1_phib2_5e-05_w2_0.0_xn_120_nx_4097_lambda_0.007_f_0.08.pkl'
# list_dir = 'teste_artigo1_phib2_7e-05_w2_0.0_xn_120_nx_4097_lambda_0.007_f_0.3.pkl'
# list_dir = 'phib2_1e-06_w2_0.0_xn_60_nx_4097_lambda_0.007_f_1.0_v_50.pkl'
# list_dir = [[0, list_dir]]

print(list_dir)

y_list = []
e_list = []
x_list = []
t_list = []
c_neg = []

params = []
for i in list_dir:
    f = open(diretorio+i[1], 'rb')
    l = pickle.load(f)
    x_list.append(l[0]/l[2]['xn'])
    # x_list.append(l[0] / 1)
    y_list.append(l[1][0::2])
    e_list.append(l[1][1::2])
    c_neg.append((l[-1]['c_salt'] + l[-1]['f']*l[-1]['phib2'])*np.exp(l[1][0::2]))
    i[1] = i[1].split('.pk')[0]
    temp = str(i[1].split('_')[atributo])
    if temp == 'phib2':
        t_list.append('$\phi_b^2$'
                      +' = '+
                      str(i[1].split('_')[atributo+1]))
    else:
        t_list.append(str(i[1].split('_')[atributo])
                      + ' = ' +
                      str(i[1].split('_')[atributo + 1]))
    params.append(l[-1])
    f.close()

l = list_dir[0][1]
l = l.split('.pk')[0]
l = l.split('_')
l2 = l[0:atributo]+l[atributo+2:]
l = str.join('_', l2)

red = np.linspace(1.,0., len(list_dir))
green = np.linspace(0,0., len(list_dir))
blue = np.linspace(0.,1., len(list_dir))
linestyle2 = ['solid', 'dashed', 'dashdot', (0, (6, 2, 6, 4))]
fig, ax = plt.subplots()

# colours = []
# for i in range(len(x_list)):
#     colours.append('#'+str(hex(15-i)[-1])+'0'+str(hex(i)[-1]))

size = 30
for i in range(len(x_list)):
    plt.plot(x_list[i], y_list[i], color = (red[i], green[i], blue[i]), linestyle = linestyle2[i%4])
plt.xlabel('$x/x_n$', fontsize=size)
plt.ylabel('y', fontsize=size, rotation=0, labelpad=35)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.subplots_adjust(left=0.18, right=0.85, bottom=0.18, top=0.85)
# plt.xlim(right = 0.6)
plt.show()

for i in range(len(x_list)):
    plt.plot(x_list[i], e_list[i]**2, color = (red[i], green[i], blue[i]), linestyle = linestyle2[i%4])
# plt.plot(x_list[0], eta_p, 'g')
# plt.plot( x_list[0], np.ones((len(x_list[0]), 1)), color = (0, 0.5, 0))
plt.xlabel('$x/x_n$', fontsize=size)
plt.ylabel('$\eta^2$', fontsize=size, rotation=0, labelpad=35)
plt.subplots_adjust(left=0.18, right=0.85, bottom=0.18, top=0.85)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
# plt.xlim(right = 0.6)
plt.show()

size = 20
plt.title(l, fontsize=size)
for i in range(len(x_list)):
    plt.plot(x_list[i], y_list[i], color = (red[i], green[i], blue[i]), linestyle = linestyle2[i%4])
plt.legend(t_list, fontsize=size)
plt.xlabel('$x/x_n$', fontsize=size)
plt.ylabel('y', fontsize=size, rotation=0, labelpad=35)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
plt.subplots_adjust(left=0.18, right=0.85, bottom=0.18, top=0.85)
plt.show()

plt.title(l, fontsize=size)
for i in range(len(x_list)):
    plt.plot(x_list[i], e_list[i]**2, color = (red[i], green[i], blue[i]), linestyle = linestyle2[i%4])
# plt.plot(x_list[0], eta_p, 'g')
# plt.plot( x_list[0], np.ones((len(x_list[0]), 1)), color = (0, 0.5, 0))
plt.xlabel('$x/x_n$', fontsize=size)
plt.ylabel('$\eta^2$', fontsize=size, rotation=0, labelpad=35)
plt.legend(t_list, fontsize=size)
plt.xticks(fontsize=size)
plt.yticks(fontsize=size)
# plt.savefig(diretorio +'2.pdf')
plt.show()

# plt.title(l, fontsize=size)
# for i in range(len(x_list)):
#     plt.plot(x_list[i], c_neg[i]*1000, color = (red[i], green[i], blue[i]))
# # plt.plot( x_list[0], np.ones((len(x_list[0]), 1)), color = (0, 0.5, 0))
# plt.xlabel('x/D', fontsize=size)
# plt.ylabel('$c_{-}(x)(x1000)$', fontsize=size, rotation=90, labelpad=35)
# plt.legend(t_list, fontsize=size)
# plt.xticks(fontsize=size)
# plt.yticks(fontsize=size)
# plt.show()

# plt.title('teste solucao exata', fontsize=size)
# plt.plot(x_list[0], eta_p, color = (red[0], green[0], blue[0]))
# # plt.plot( x_list[0], np.ones((len(x_list[0]), 1)), color = (0, 0.5, 0))
# plt.xlabel('x/D', fontsize=size)
# plt.ylabel('$\eta^2_p$', fontsize=size, rotation=0, labelpad=35)
# plt.legend(t_list, fontsize=size)
# plt.xticks(fontsize=size)
# plt.yticks(fontsize=size)
# plt.show()