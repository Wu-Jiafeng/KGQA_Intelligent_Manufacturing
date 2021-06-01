import ujson as json
from py2neo import Graph, Node, Relationship #,NodeSelector
from neo_db.config import graph,e2t

def importt(filename):
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            rela_array = line.strip("\n").split(",")
            print(rela_array)
            graph.run("MERGE(p: Person{cate:'%s',Name: '%s'})" % (rela_array[3], rela_array[0]))
            graph.run("MERGE(p: Person{cate:'%s',Name: '%s'})" % (rela_array[4], rela_array[1]))
            graph.run(
                "MATCH(e: Person), (cc: Person) \
                WHERE e.Name='%s' AND cc.Name='%s'\
                CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
                RETURN r" % (rela_array[0], rela_array[1], rela_array[2], rela_array[2])

            )

def importtxt(filename):
    #e2t=json.load(open("../triple_data/e2t.json", "r", encoding="utf-8-sig"))
    with open(filename, encoding="utf-8") as f:
        for line in f.readlines():
            subj,pred,obj = line.replace("\ufeff","").strip("\n").split("\t")
            subj_t,obj_t=e2t[subj],e2t[obj]
            print(subj,subj_t,pred,obj,obj_t)
            graph.run("MERGE(p: "+subj_t+"{ Name: '%s',type:'%s'})" % (subj,subj_t))
            graph.run("MERGE(p: "+obj_t+"{ Name: '%s',type:'%s'})" % (obj,obj_t))
            graph.run(
                "MATCH(e: "+subj_t+"), (cc: "+obj_t+") \
                WHERE e.Name='%s' AND cc.Name='%s'\
                CREATE(e)-[r:%s{relation: '%s'}]->(cc)\
                RETURN r" % (subj, obj, pred, pred)
            )

if __name__=="__main__":
    #importt("../triple_data/relation.txt")
    importtxt("../triple_data/output.txt")