import pickle
import bson

print "decoding resume bson file"

resume_file = open('resume_1_million/zippia/resume_1_million.bson').read()
decoded_resume_list = bson.decode_all(resume_file)

print "first element: ", decoded_resume_list[0]

with open('decoded_resume_list.pickle', 'wb') as handle:
    pickle.dump(decoded_resume_list, handle)

print "decoding soc bson file"

soc_file = open('soc_4_million/zippia/soc_4_million.bson').read()
decoded_soc_list = bson.decode_all(soc_file)

print decoded_soc_list[0]

with open('decoded_soc_list.pickle', 'wb') as handle:
    pickle.dump(decoded_soc_list, handle)
