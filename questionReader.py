import json
import re

class questionReader:
    def __init__(self, inputFileName):
        self.inputFN = inputFileName
        print "Initializing"
        self.jString = json.loads(open(inputFileName,'r').read())
        self.jString = self.jString['questions']
        self.questionLen = len(self.jString)
        self.questionIndexMap = {}
        self.answerIndexMap = {}
        self.preprocessed = False
        self.qIndex = 0
        print "Initializing done"
    
    def preprocess(self, test=False):
        
        print "Start preprocessing"
        
        wordIndex = 0
        ansIndex = 0
        maxSentenceLen = 0
        
        for el in self.jString:
            question = re.sub('[\"\?]', '', el['question'].lower())
            answer = el['answer']
            
            # put answer in the answer map if it's not already in it
            if answer not in self.answerIndexMap : 
                self.answerIndexMap[answer] = ansIndex
                ansIndex += 1
            
            # put question words in the map if it's not already in the map.
            # we also assign a unique integer to each question word.
            splits = question.split()
            if len(splits) > maxSentenceLen : maxSentenceLen = len(splits)
            for word in splits :
                if word not in self.questionIndexMap :
                    self.questionIndexMap[word] = wordIndex
                    wordIndex += 1
                    
        # preprocessing done            
        print "Preprocessing done. qVocabSize: %d, aVocabSize: %d, maxSentenceLen: %d" % (wordIndex, len(self.answerIndexMap), maxSentenceLen)
        self.preprocessed = True
        
        return self.questionIndexMap, self.answerIndexMap
        
    def readNextPair(self, test=False):
        if not self.preprocess :
            print "questionReader.preprocess must procede reading input data. Exiting."
            exit(1)
        
        if test :
            cur_pair = self.jString[self.qIndex]
            self.qIndex = self.qIndex + 1
            imageFN = cur_pair['image_filename']
            question = cur_pair['question']
            
            return imageFN, question
        
        else :
            cur_pair = self.jString[self.qIndex]
            self.qIndex = self.qIndex + 1
            doneReading = (self.qIndex == self.questionLen)
            if doneReading : self.qIndex = 0
            imageFN = cur_pair['image_filename']
            question = cur_pair['question']
            answer = cur_pair['answer']
        
            return imageFN, question, answer, doneReading
'''
qr = questionReader('')
qr.preprocess()
for i in range(5):
    imageFN, question, answer =  qr.readNextPair()
    print imageFN, question, answer
    break
'''
