import pandas as pd
import json
import SPARQLWrapper
from SPARQLWrapper import SPARQLWrapper, JSON
import numpy as np
from tqdm import tqdm
import rdflib
pd.set_option('display.max_rows', 100)


#  SPARQL queries for more then 10,000 triples 
def get_sparql_dataframe(service, query):
    sparql = SPARQLWrapper(service)
    out = []
    num_iter = 20
    
    for i in range(num_iter):
        sparql.setQuery(query + " OFFSET " + str(i) + "000")
        sparql.setReturnFormat(JSON)
        result = sparql.query()
        processed_results = json.load(result.response)
        cols = processed_results['head']['vars']

        for row in processed_results['results']['bindings']:
            item = []
            for c in cols:
                item.append(row.get(c, {}).get('value'))
            out.append(item)
        
    return pd.DataFrame(out, columns=cols)


def find_differences(t1,t2): 
    t1 = t1.reset_index(drop=True)
    t2 = t2.reset_index(drop=True)
    
    df = pd.concat([t1, t2], keys=['t1', 't2'])
    df = df.drop(['s'], axis=1)
    df_gpby = df.groupby(list(df.columns))
    idx = [x[0] for x in df_gpby.groups.values() if len(x) == 1]
    
    return df.reindex(idx).sort_index()

def filter_predicates(some_list):
    
    filter_outs = ['http://www.w3.org/2000/01/rdf-schema#label',
                   'https://data.cooperationdatabank.org/vocab/prop/meanContributionOrWithdrawalForCondition',
                   'https://data.cooperationdatabank.org/vocab/prop/nCondition',
                   'https://data.cooperationdatabank.org/vocab/prop/sDforCondition', 
                   'https://data.cooperationdatabank.org/vocab/prop/proportionOfCooperationCondition', 
                   'https://data.cooperationdatabank.org/vocab/prop/individualDifferenceLevel'] 
    
    for uri in filter_outs: 
        if uri in some_list: 
            some_list.remove(uri)
            #  while thing in some_list: some_list.remove(thing)   
    
    return some_list 

#  SPARQL queries for less then 10,000 triples 
def get_sparql_dataframe2(service, query):
    sparql = SPARQLWrapper(service)
    out = []
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    result = sparql.query()
    processed_results = json.load(result.response)
    cols = processed_results['head']['vars']

    for row in processed_results['results']['bindings']:
        item = []
        for c in cols:
            item.append(row.get(c, {}).get('value'))
        out.append(item)
        
    return pd.DataFrame(out, columns=cols)

def do_the_thing(observations): 
    outcome = pd.DataFrame()
    loop = tqdm(total = len(observations), position=0, leave=False)
    for i in range(len(observations)): # len(observations)
        row = observations.loc[i]
        observation = row['obsLabel']
        effect_size = row['ES']

        treatments = {'t1': outcome, 't2': outcome}

        for treatment in treatments: 
    #         print(str(row[treatment]))

            query = """
                        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                        PREFIX class: <https://data.cooperationdatabank.org/vocab/class/>

                        CONSTRUCT { <""" + str(row[treatment]) + """> ?pred ?obj } WHERE {
                         <""" + str(row[treatment]) + """> ?pred ?obj . 
    #                    ?pred rdfs:range ?propVarClass . 
    #                    ?propVarClass rdfs:subClassOf ?subClassIndependent . 
    #                    ?subClassIndependent rdfs:subClassOf ?IndependentClass .                    
                        } 
                        """

            treatments[treatment] = get_sparql_dataframe2(url, query)

        #  Show percentage bar
        loop.set_description("Loading...".format(i))
        loop.update(1)

        df_differences = find_differences(treatments['t1'].sort_values(by=['p']) , treatments['t2'].sort_values(by=['p']))
        different_predicates = df_differences['p'].unique().tolist()
        independent_variables = filter_predicates(different_predicates)
        obs_index = i
        observations.at[obs_index, 'independents'] = independent_variables
    loop.close()