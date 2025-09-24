import pickle

def load_pickle_data(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

print("Starting test_load.py")

try:
    X_train = load_pickle_data("data/train.pickle")
    print(f"Loaded X_train, shape/type: {type(X_train)}, ", end='')
    try:
        print(f"shape = {X_train.shape}")
    except:
        print("no shape attribute")

except Exception as e:
    print("Error loading X_train:", e)

print("test_load.py finished")
