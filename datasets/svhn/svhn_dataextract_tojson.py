# SVHN extracts data from the digitStruct.mat full numbers files.  The data can be downloaded
# the Street View House Number (SVHN)  web site: http://ufldl.stanford.edu/housenumbers.  
#
# This is an A2iA tweak (YG -9 Jan 2014) of the script found here :
# http://blog.grimwisdom.com/python/street-view-house-numbers-svhn-and-octave
#
# The digitStruct.mat files in the full numbers tars (train.tar.gz, test.tar.gz, and extra.tar.gz) 
# are only compatible with matlab.  This Python program can be run at the command line and will generate 
# a json version of the dataset.
#
# Command line usage:
#       SVHN_dataextract.py [-f input] [-o output_without_extension]
#    >  python SVHN_dataextract.py -f digitStruct.mat -o digitStruct
#
# Issues:
#    The alibility to split in several files has been removed from the original
#    script.
#

import h5py
import optparse
from json import JSONEncoder

parser = optparse.OptionParser()
parser.add_option("-f", dest="fin", help="Matlab full number SVHN input file", default="digitStruct.mat")
parser.add_option("-o", dest="filePrefix", help="name for the json output file", default="digitStruct")
(options,args)= parser.parse_args()

fin = options.fin

# The DigitStructFile is just a wrapper around the h5py data.  It basically references 
#    inf:              The input h5 matlab file
#    digitStructName   The h5 ref to all the file names
#    digitStructBbox   The h5 ref to all struc data
class DigitStructFile:
    def __init__(self, inf):
        self.inf = h5py.File(inf, 'r')
        self.digitStructName = self.inf['digitStruct']['name']
        self.digitStructBbox = self.inf['digitStruct']['bbox']

# getName returns the 'name' string for for the n(th) digitStruct. 
    def getName(self,n):
        return ''.join([chr(c[0]) for c in self.inf[self.digitStructName[n][0]].value])

# bboxHelper handles the coding difference when there is exactly one bbox or an array of bbox. 
    def bboxHelper(self,attr):
        if (len(attr) > 1):
            attr = [self.inf[attr.value[j].item()].value[0][0] for j in range(len(attr))]
        else:
            attr = [attr.value[0][0]]
        return attr

# getBbox returns a dict of data for the n(th) bbox. 
    def getBbox(self,n):
        bbox = {}
        bb = self.digitStructBbox[n].item()
        bbox['height'] = self.bboxHelper(self.inf[bb]["height"])
        bbox['label'] = self.bboxHelper(self.inf[bb]["label"])
        bbox['left'] = self.bboxHelper(self.inf[bb]["left"])
        bbox['top'] = self.bboxHelper(self.inf[bb]["top"])
        bbox['width'] = self.bboxHelper(self.inf[bb]["width"])
        return bbox

    def getDigitStructure(self,n):
        s = self.getBbox(n)
        s['name']=self.getName(n)
        return s

# getAllDigitStructure returns all the digitStruct from the input file.     
    def getAllDigitStructure(self):
        return [self.getDigitStructure(i) for i in range(len(self.digitStructName))]

# Return a restructured version of the dataset (one structure by boxed digit).
#
#   Return a list of such dicts :
#      'filename' : filename of the samples
#      'boxes' : list of such dicts (one by digit) :
#          'label' : 1 to 9 corresponding digits. 10 for digit '0' in image.
#          'left', 'top' : position of bounding box
#          'width', 'height' : dimension of bounding box
#
# Note: We may turn this to a generator, if memory issues arise.
    def getAllDigitStructure_ByDigit(self):
        pictDat = self.getAllDigitStructure()
        result = []
        structCnt = 1
        for i in range(len(pictDat)):
            item = { 'filename' : pictDat[i]["name"] }
            figures = []
            for j in range(len(pictDat[i]['height'])):
               figure = {}
               figure['height'] = pictDat[i]['height'][j]
               figure['label']  = pictDat[i]['label'][j]
               figure['left']   = pictDat[i]['left'][j]
               figure['top']    = pictDat[i]['top'][j]
               figure['width']  = pictDat[i]['width'][j]
               figures.append(figure)
            structCnt = structCnt + 1
            item['boxes'] = figures
            result.append(item)
        return result

dsf = DigitStructFile(fin)
dataset = dsf.getAllDigitStructure_ByDigit()
fout = open(options.filePrefix + ".json",'w')
fout.write(JSONEncoder(indent = True).encode(dataset))
fout.close()

