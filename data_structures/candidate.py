class Candidate:
 def __init__(self, id, description, is_female):
    self.id = id
    self.description = description
    self.is_female = is_female
    
if __name__ == "__main__" : 
    print("running candidate.py")
else :
    print("imported ", __name__)