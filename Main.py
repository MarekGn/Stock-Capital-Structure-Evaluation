from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import DataTools.DataLoader as dt
from Network.DNN import DNN, load_network
from GA.Population import Population
from tqdm import tqdm
from Logger.Logger import Logger
import numpy as np

def train_network(batchSize=8, epochs=2000):
    ############### DATA ###############
    # LOAD DATA
    X_train, Y_train = dt.load_all_csv_files()
    # SCALE DATA
    scalerX = MinMaxScaler(feature_range=(0, 1))
    scalerY = MinMaxScaler(feature_range=(0, 1))
    X_train = scalerX.fit_transform(X_train)
    Y_train = scalerY.fit_transform(Y_train.reshape((-1, 1)))
    # SAVE SCALERS
    dt.save_scalers(scalerX, scalerY)
    # SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, shuffle=True)
    print("TrainX {}    TrainY {}   TestX {}    TestY {}".format(len(X_train), len(y_train), len(X_test), len(y_test)))
    ############### NETWORK ###############
    # CREATE NETWORK
    dnn = DNN()
    # CONFIGURE NETWORK
    dnn.create_model(55)
    dnn.configure()
    # TRAIN NETWORK
    dnn.train(x=X_train, y=y_train, valX=X_test, valY=y_test, batchSize=batchSize, epochs=epochs)


def genetic_training(iterations=100, pop_size=100, idv_size=55, mutation_probability=0.02, alpha=1.3):
    scalerX, scalerY = dt.load_scalers()
    dnn = load_network("trainval")
    pop = Population(pop_size=pop_size, idv_size=idv_size, scalerX=scalerX, scalerY=scalerY)
    logger = Logger()
    for _ in tqdm(range(iterations), desc="Iterations Progress: "):
        pop.cal_fitness(dnn)
        logger.log_idv.append(pop.pop[0].genome)
        logger.save_best_genome("best_idv_history_pop{}_mut{}_alpha{}3attempt".format(pop_size, mutation_probability, alpha))
        pop.cal_probability(alpha=alpha)
        pop.get_new_pop()
        pop.mutate_population(mutation_probability=mutation_probability)

if __name__ == '__main__':
    # train_network(batchSize=16, epochs=5000)
    genetic_training(iterations=10000, pop_size=250, idv_size=55, mutation_probability=0.02, alpha=1.25)






