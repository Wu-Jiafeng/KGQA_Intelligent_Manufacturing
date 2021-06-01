from neo_db.config import graph, CA_LIST, similar_words,e2t
from spider.show_profile import get_profile
import codecs
import os
import ujson as json
import base64

def query(name):
    data = graph.run(
    "match(p )-[r]->(n:"+e2t[name]+"{Name:'%s'}) return  p.Name,p.type,r.relation,n.Name,n.type\
        Union all\
    match(p:"%(name)+e2t[name]+" {Name:'%s'}) -[r]->(n) return p.Name,p.type, r.relation, n.Name,n.type" % (name)
    )
    data = list(data)
    return get_json_data(data)

def get_json_data(data):
    json_data={'data':[],"links":[]}
    d=[]

    if len(data)==0: return json_data
    
    for i in data:
        # print(i["p.Name"], i["r.relation"], i["n.Name"], i["p.cate"], i["n.cate"])
        d.append(i['p.Name']+"_"+i['p.type'])
        d.append(i['n.Name']+"_"+i['n.type'])
        d=list(set(d))
    name_dict={}
    count=0
    for j in d:
        j_array=j.split("_")
    
        data_item={}
        name_dict[j_array[0]]=count
        count+=1
        data_item['name']=j_array[0]
        data_item['category']=CA_LIST[j_array[1]]
        json_data['data'].append(data_item)
    for i in data:
   
        link_item = {}
        
        link_item['source'] = name_dict[i['p.Name']]
        
        link_item['target'] = name_dict[i['n.Name']]
        link_item['value'] = i['r.relation']
        json_data['links'].append(link_item)

    return json_data
# f = codecs.open('./static/test_data.json','w','utf-8')
# f.write(json.dumps(json_data,  ensure_ascii=False))

def get_KGQA_answer(entities,array):
    data_array,s_array=[],set()
    for word in array:
        if word in similar_words.keys(): s_array.add(word)

    for entity in entities:
        name,etype=entity["word"],entity["type"]
        for i ,word in enumerate(s_array):
            if i != 0 and len(data_array)!=0: name=data_array[-1]['p.Name']
            data = graph.run(
                "match(p)-[r:%s{relation: '%s'}]->(n:%s{Name:'%s'}) return  p.Name,n.Name,r.relation,p.type,n.type" % (
                    similar_words[word], similar_words[word], etype,name)
            )
            data = list(data)
            print(data)
            data_array.extend(data)
            data = graph.run(
                "match(p:%s{Name:'%s'})-[r:%s{relation: '%s'}]->(n) return  p.Name,n.Name,r.relation,p.type,n.type" % (
                    etype, name,similar_words[word], similar_words[word])
            )
            data = list(data)
            print(data)
            data_array.extend(data)

        print("==="*36)

    if len(data_array)==0: image_name="./spider/images/None.jpg"
    else: image_name="./spider/images/"+"%s.jpg" % (str(data_array[-1]['p.Name']))
    if not os.path.exists(image_name): image_name="./spider/images/None.jpg"
    with open(image_name, "rb") as image:
            base64_data = base64.b64encode(image.read())
            b=str(base64_data)

    if len(data_array)==0: pname='数控纺织机械'
    else: pname=str(data_array[-1]['p.Name'])
    return [get_json_data(data_array), get_profile(pname), b.split("'")[1]]

def get_answer_profile(name):
    image_name = "./spider/images/" + "%s.jpg" % (str(name))
    if not os.path.exists(image_name): image_name = "./spider/images/None.jpg"
    with open(image_name, "rb") as image:
        base64_data = base64.b64encode(image.read())
        b = str(base64_data)
    return [get_profile(str(name)), b.split("'")[1]]

if __name__=="__main__":
    data = graph.run("match(p )-[r]->(n) return  p.Name,p.type,r.relation,n.Name,n.type")
    data = list(data)
    json_data=get_json_data(data)
    json.dump(json_data,open("../static/data.json",'w',encoding="utf-8-sig"),ensure_ascii=False)

        



