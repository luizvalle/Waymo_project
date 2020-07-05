from google.cloud import storage
from glob import glob

TFRECORDS_DIRECTORY = "/home/luizvalle/Documents/github/Waymo_project/datasets/training_00003/"

def main():
    store_directory = classify_tfrecords(TFRECORDS_DIRECTORY)

    bucket = get_bucket()

    for tfrecord, store_path in store_directory.items():
        tfrecord_name = tfrecord[tfrecord.rfind("/") + 1:]
        blob = bucket.blob(f"{store_path}{tfrecord_name}")
        print(f"Uploading {tfrecord_name}")
        blob.upload_from_filename(tfrecord)


def classify_tfrecords(directory):
    store_directory = dict()
    tfrecords = glob(directory + "*.tfrecord")
    for tfrecord in tfrecords:
        tfrecord_name = tfrecord[tfrecord.rfind("/") + 1:]
        classification = prompt_classification(tfrecord_name)
        if classification == "skip":
            continue
        store_directory[tfrecord] = get_path_from_classification(classification)
    return store_directory

def prompt_classification(tfrecord_name):
    print(tfrecord_name)
    response = input("\tCategory (df, dl, hf, hl): ").lower()
    while response not in ["df", "dl", "hf", "hl", "skip"]:
        response = input("\tCategory (df, dl, hf, hl): ").lower()
    return response

def get_bucket():
    storage_client = storage.Client.from_service_account_json("sumo-ns3-gpu-ae3ec90a4aa7.json")
    return storage_client.get_bucket("waymo_data")

def get_path_from_classification(user_input):
    path = "Waymo_originData/"
    
    if user_input[0] == "d":
        path += "Downtown/"
    elif user_input[0] == "h":
        path += "Highway/"

    if user_input[1] == "f":
        path += "Carfollowing/"
    elif user_input[1] == "l":
        path += "Lanechange/"

    return path

if __name__ == "__main__":
    main()