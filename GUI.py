import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
import Run
sg.change_look_and_feel('DarkTeal12')    # look and feel theme


# Designing layout
layout = [[sg.Text("\t\t\tSelect_dataset   "),sg.Combo(['db1','db2', 'db3']),sg.Text("\n")],
          [sg.Text("\t\t"),sg.Combo(["cluster size"], size=(5, 2)), sg.Text(""), sg.InputText(size=(20, 20), key='1'),sg.Button("START", size=(10, 2))],[sg.Text('\n')],
          [sg.Text("\t\t  Method1 \t\tMethod2   \t\t  Method3\t\t     Method4")],
          [sg.Text('\tNCEI'), sg.In(key='11',size=(20,20)), sg.In(key='12',size=(20,20)), sg.In(key='13',size=(20,20)), sg.In(key='14',size=(20,20)),sg.Text("\n")],
          [sg.Text('\tARI'), sg.In(key='21',size=(20,20)), sg.In(key='22',size=(20,20)), sg.In(key='23',size=(20,20)), sg.In(key='24',size=(20,20)), sg.Text("\n")],
          [sg.Text('\tFM'), sg.In(key='31', size=(20, 20)), sg.In(key='32', size=(20, 20)),sg.In(key='33', size=(20, 20)), sg.In(key='34', size=(20, 20)),sg.Text("\n")],
          [sg.Text('\tJI'), sg.In(key='41', size=(20, 20)), sg.In(key='42', size=(20, 20)),
           sg.In(key='43', size=(20, 20)), sg.In(key='44', size=(20, 20)), sg.Text("\n")],
          [sg.Text('\tMI'), sg.In(key='51', size=(20, 20)), sg.In(key='52', size=(20, 20)),
           sg.In(key='53', size=(20, 20)), sg.In(key='54', size=(20, 20)), sg.Text("\n")],
          [sg.Text('\tHI'), sg.In(key='61', size=(20, 20)), sg.In(key='62', size=(20, 20)),
           sg.In(key='63', size=(20, 20)), sg.In(key='64', size=(20, 20)), sg.Text("\n")],
          [sg.Text('\t\t\t\t\t\t\t\t\t\t\t '), sg.Button('Run Graph'), sg.Button('CLOSE')]]


# to plot graphs
def plot_graph(result_1, result_2, result_3,result_4,result_5,result_6):
    plt.figure(dpi=120)
    loc, result = [], []
    result.append(result_1)  # appending the result
    result.append(result_2)
    result.append(result_3)
    result.append(result_4)
    result.append(result_5)
    result.append(result_6)
    print(result)

    result = np.transpose(result)

    # labels for bars
    labels = ['K-means','FCM','FLICM','DFC']  # x-axis labels ############################
    tick_labels = ['NCEI','ARI','FM','JI','MI','HI']  #### metrics
    bar_width, s = 0.15, 0.025  # bar width, space between bars

    for i in range(len(result)):  # allocating location for bars
        if i == 0:  # initial location - 1st result
            tem = []
            for j in range(len(tick_labels)):
                tem.append(j + 1)
            loc.append(tem)
        else:  # location from 2nd result
            tem = []
            for j in range(len(loc[i - 1])):
                tem.append(loc[i - 1][j] + s + bar_width)
            loc.append(tem)

    # plotting a bar chart
    for i in range(len(result)):
        plt.bar(loc[i], result[i], label=labels[i], tick_label=tick_labels, width=bar_width)

    plt.legend(loc=(0.25, 0.25))  # show a legend on the plot -- here legends are metrics
    plt.show()  # to show the plot



# Create the Window layout
window = sg.Window('GUI', layout)

# event loop
while True:
    event, values = window.read()  # displays the window
    if event == "START":

        tp = int(values['1'])
        Dataset, tr_per = values[0], tp  # Dataset, Training percentage, No. of Feature Selection
        dts = Dataset
        print("Running......")
        NCEI,ARI,FM,JI,MI,HI = Run.callmain(dts,tp)

        window.Element('11').Update(NCEI[0])
        window.Element('12').Update(NCEI[1])
        window.Element('13').Update(NCEI[2])
        window.Element('14').Update(NCEI[3])

        window.Element('21').Update(ARI[0])
        window.Element('22').Update(ARI[1])
        window.Element('23').Update(ARI[2])
        window.Element('24').Update(ARI[3])

        window.Element('31').Update(FM[0])
        window.Element('32').Update(FM[1])
        window.Element('33').Update(FM[2])
        window.Element('34').Update(FM[3])

        window.Element('41').Update(JI[0])
        window.Element('42').Update(JI[1])
        window.Element('43').Update(JI[2])
        window.Element('44').Update(JI[3])

        window.Element('51').Update(MI[0])
        window.Element('52').Update(MI[1])
        window.Element('53').Update(MI[2])
        window.Element('54').Update(MI[3])

        window.Element('61').Update(HI[0])
        window.Element('62').Update(HI[1])
        window.Element('63').Update(HI[2])
        window.Element('64').Update(HI[3])



        print("\n Done.")
    if event == 'Run Graph':
        plot_graph(NCEI, ARI, FM, JI, MI, HI)

    if event == 'CLOSE':
        break
        window.close()
