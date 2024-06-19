import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import support_fn


def plot_all():

      SMALL_SIZE = 8
      MEDIUM_SIZE = 10
      BIGGER_SIZE = 12

      plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
      plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
      plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
      plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
      plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
      plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
      plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
      plt.rcParams['axes.labelweight'] = 'bold'


      graphpath="out_graphs_csv_p1"
      full_gpath=os.path.realpath(graphpath)
      if not os.path.exists(full_gpath):
          os.makedirs(full_gpath)

      with open('result_1b.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
          result_all= pickle.load(f)

      legend_str=['K-means','FCM','FLICM','DFC']
      xi=[3,4,5,6]

      #--- comparitive analysis 1 ---
      #----------  comparitive  1    -     accuracy ------------
      index = np.arange(1, 5, 1)
      bar_width = 0.14
      opacity = 0.8
      xpos = np.array([1, 2, 3,4]) + 2 * bar_width
      data=result_all['n1']

      data =np.array(data)
      plt.figure(0)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('NCEI')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'NCEI1.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'NCEI1.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['m1']

      data =np.array(data)
      plt.figure(1)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Mutual Information')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'MI1.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'MI1.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['a1']
      data =np.array(data)
      plt.figure(2)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Adjusted Rand Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'ari1.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'ari1.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['f1']
      data =np.array(data)
      plt.figure(3)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Fowlkes-Mallows Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'fm1.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'fm1.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['j1']
      data =np.array(data)
      plt.figure(4)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Jaccard Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'j1.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'j1.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['h1']
      data =np.array(data)
      plt.figure(5)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Hubert Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'h1.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'h1.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)


      ##################db2
      data=result_all['n2']

      data =np.array(data)
      plt.figure(6)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('NCEI')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'NCEI2.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'NCEI2.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['m2']

      data =np.array(data)
      plt.figure(7)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Mutual Information')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'MI2.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'MI2.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['a2']
      data =np.array(data)
      plt.figure(8)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Adjusted Rand Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'ari2.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'ari2.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['f2']
      data =np.array(data)
      plt.figure(9)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Fowlkes-Mallows Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'fm2.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'fm2.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)
      #
      data=result_all['j2']
      data =np.array(data)
      plt.figure(10)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Jaccard Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'j2.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'j2.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)

      #
      data=result_all['h2']
      data =np.array(data)
      plt.figure(11)
      plt.bar(index, data[:, 0], bar_width, alpha=opacity,edgecolor ='black', color='#BB1181')
      plt.bar(index + 1 * bar_width, data[:, 1], bar_width, alpha=opacity,edgecolor ='black', color='#7ABC0F')
      plt.bar(index + 2 * bar_width, data[:, 2], bar_width, alpha=opacity,edgecolor ='black', color='#0084C6')
      plt.bar(index + 3 * bar_width, data[:, 3], bar_width, alpha=opacity,edgecolor ='black', color='#c91712')
      ax=plt.gca()
      ax.set_xticks(xpos)
      ax.set_xticklabels(xi)
      ax.legend(legend_str)
      plt.xlabel('Cluster size')
      plt.ylabel('Hubert Index')
      plt.legend(legend_str,loc='upper center', bbox_to_anchor=(0.48, 1.25),
                 fancybox=True, ncol=2,fontsize='10')

      plt.tight_layout()
      plt.savefig(os.path.join(full_gpath,'h2.png'), format='png', dpi=300)
      # plt.title('Key complexity')
      csv_name=os.path.join(full_gpath,'h2.csv')
      support_fn.write_to_file2(data,csv_name,legend_str)




      plt.show()


if __name__ == '__main__':
    plot_all()


