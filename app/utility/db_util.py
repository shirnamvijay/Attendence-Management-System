import pymongo

class DBUtil:

    def __init__(self):
        self.client = pymongo.MongoClient('localhost', 27017)
        self.db = self.client.get_database('AMS_DB')

    def insert(self, collection, data):
        collection = self.db[collection]
        return collection.insert_one(data)

    def get_all(self, collection):
        collection = self.db[collection]
        return collection.find()

    def check_exist(self, collection, data):
        collection = self.db[collection]
        return collection.find(data)

    def count(self, collection, data={}):
        collection = self.db[collection]
        return collection.count_documents(data)

    def get_last(self, collection, data={}):
        collection = self.db[collection]
        return list(collection.find(data))[-1]

    def close(self):
        self.client.close()