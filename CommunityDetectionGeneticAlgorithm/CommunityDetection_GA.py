#%%
import networkx as nx
import numpy as np
import scipy as sp
import random
#%%

#%%
def create_population(m,A):
    # creare populatie initiala
    # fiecare cromozom este un array de lungimea nodurilor grafului
    # e SIGUR doar daca pe pozitia j (sursa) si valoarea din j (destinatia) exista (sunt conectate)
    # si in matricea de adiacenta a grafului
    n = A.shape[0] # no of rows
    population= np.random.randint(0,n,(m,n)) #genereaza m cromozmi

    for i in range(m):
        for j in range(n):
            #verificam daca un cromozm este SIGUR (i.e „muchiile (corespunzatoare reprezentarii cromozomului) in graf sunt conectate”)

            d = population[i,j]
            if (A[j,d] != 1):
                # dacă nu e safe, atunci ”îl facem safe”
                neighs = []
                for k in range(n):
                    if (A[j,k]!=0): neighs.append(k)

                pos = random.randint(0,len(neighs)-1)
                population[i,j] = neighs[pos]

    return population
def greedyCommunitiesDetection(G):
    # algoritm de calcularea modularitatii pentru Fast Newman (comparisom reasons)
    c = list(nx.algorithms.community.modularity_max.greedy_modularity_communities(G))
    mm_G = []
    for i,j in enumerate(c):
        a = list(j)
        mm_G.append(a)
    per = nx.algorithms.community.quality.modularity(G,mm_G)
    num_G = len(mm_G)
    return  num_G, per
#%%
def find_connected_subgraphs(G):
    #subgrafurile din G
    return list(G.subgraph(c) for c in nx.connected_components(G))

def find_fitness(population,r,A):
    # Calculate fitness of chromosomes
    m,n = population.shape
    fits = []
    for i in range(m):
        #matricea de adiacenta a cromozomului
        adj = np.zeros((n,n))
        for j in range(n):
            adj[j,int(population[i,j])] = 1
            adj[int(population[i,j]),j] = 1
        # graful cromozomului
        G = nx.from_numpy_array(adj)
        #subgrafurile din cromozom
        subgraphs = find_connected_subgraphs(G)
        #comunitatile formate ca atare din subgrafuri
        coms = [list(subgraph.nodes()) for subgraph in subgraphs]

        # Formula 1: Fitness bazat pe formula din Clara Pizzuti, 2008 `GA-Net`
        # extra info in articol
        CS = 0
        for com in coms:
            row_idx = np.array(com)
            col_idx =  np.array(com)
            #submatricea S = (I,J) unde I un sub-set de randuri din A si J subset de coloane
            # din A;
            # vs = sum(sum(a_ij)) - adica numarul de 1 din aceasta submatrice
            sub = A[row_idx[:, None], col_idx]
            vs = sum(sum(sub))
            # power mean of S of order r -> M(S)
            # (suma valoriilor medii fiecarui rand din S)^r / nr de randuri
            M = sum(np.power(np.mean(sub,axis=1),r))/len(sub)

            CS = CS + M*vs

        fits.append(CS)

    return fits
def find_fitness2(population, GG):
    # Calculate fitness of chromosomes
    m,n = population.shape
    fits = []
    for i in range(m):
        #matricea de adiacenta a cromozomului
        adj = np.zeros((n,n))
        for j in range(n):
            adj[j,int(population[i,j])] = 1
            adj[int(population[i,j]),j] = 1
        # graful cromozomului
        G = nx.from_numpy_array(adj)
        #subgrafurile din cromozom
        subgraphs = find_connected_subgraphs(G)
        #comunitatile formate ca atare din subgrafuri
        coms = [list(subgraph.nodes()) for subgraph in subgraphs]

        # Formula 2:
        Q = nx.algorithms.community.modularity(GG, coms)

        fits.append(Q)

    return fits
#%%
def proportional_selection(elite, population, fit ):
    # Alegem cu selectia ruleta (sau proportionala cu fitness-ul)
    # selectia tine cont si de cromozomii care prezinta un fitness bun ( 10% in GA-Net)
    elites = np.argsort(fit)
    new_pop = np.zeros(population.shape)

    p = []
    s = sum(fit)
    for fit_score in fit:
        # fitness-urile relative
        p.append(fit_score/s)

    for i in range(1,elite+1):
        #punem "the elites"
        pos = elites[-i]
        new_pop[i-1,:] = population[pos,:]

    for i in range(elite,population.shape[0]):
        #facem "ruleta"
        # generam random o valoare in intervalul [0, 1]
        x = random.uniform(0,1)
        k=0
        while k<population.shape[0]-1 and x > sum(p[0:k]):
            #daca "ruleta" pica in spatiul (cumulat) al fitness-ului relativ, atunci alegem drept candidat acest cromozom
            # daca nu, "ruleta merge"
            k=k+1

        new_pop[i,:] = population[k,:]

    return new_pop
#%%
def crossover(pc, population):
    # Imperechere uniforma
    # rata de imperechere (0.8) in GA-Net
    """
    Siguranta din spatele "uniform-crossover"
    We used uniform crossover because it guarantees the maintenance
of the effective connections of the nodes in the social network in the child individual.
In fact, because of the biased initialization, each individual in the population is safe,
that is it has the property, that if a gene i contains a value j, then the edge (i, j) exists.
Thus, given two safe parents, a random binary vector is created. Uniform crossover then
selects the genes where the vector is a 1 from the first parent, and the genes where the
vector is a 0 from the second parent, and combines the genes to form the child. The
child at each position i contains a value j coming from one of the two parents. Thus
the edge (i, j) exists. This implies that from two safe parents a safe child is generated
    """
    new_pop = np.zeros(population.shape)
    for i in range(population.shape[0]):
        if random.uniform(0,1) < pc:
            chroms = np.zeros((2,population.shape[1]))
            chroms[0,:] = population[i,:]
            parent2 = random.randint(0,population.shape[0]-1)
            chroms[1,:] = population[parent2,:]
            mask = np.random.randint(0,2,(population.shape[1]))

            for j in range(len(mask)):
                #offspringul (new_pop[i,j] va fi o imperechere uniforma intre
                # parent1 (chroms cu mask[j]=0)
                # si parent2 (chroms cu mask[j] = 1)
                new_pop[i,j] = chroms[mask[j],j]

        else:
            new_pop[i,:] = population[i,:]

    return new_pop
#%%
def mutation(pm, population, A):
    # Aplicam o mutatie pe un index random din cromozom, dar trebuie sa "pastram" siguranta
    # pm - rata de mutatie
    for i in range(population.shape[0]):
        if random.uniform(0,1) < pm:
            pos = random.randint(0, population.shape[1]-1)
            neighs = []
            for k in range(A.shape[1]):
                if A[pos,k] != 0: neighs.append(k)

            new_genoid = neighs[random.randint(0,len(neighs)-1)]

            population[i,pos] = new_genoid

    return population
#%%
def get_coms(chrom):
    n = len(chrom)
    adj = np.zeros((n,n))
    for j in range(n):
        adj[j,int(chrom[j])] = 1
        adj[int(chrom[j]),j] = 1

    G = nx.from_numpy_array(adj)

    subgraphs = find_connected_subgraphs(G)

    coms = [list(subgraph.nodes()) for subgraph in subgraphs]

    return coms
#%%

#%%
"""
    Parametrii:
        - G -> Graph
        - m -> marimea populatiei
        - elitism -> numarul de cromozomi "elita" (pe care sa ii adaugam la fiecare "selectie")
        - r -> exponentul r pentru functia de fitness
        - pc -> probabilitatea de "crossover"
        - pm -> probabilitatea de "mutatie"
    Output:
        - best_chrom -> cel mai bun cromozom din toate generatiile
        - coms: comunitatile detectate din acest cromozom
"""

def genetic_algorithm(G,m,elitism,r,pc,pm):
    #matricea de adiacenta
    A = nx.to_numpy_array(G)
    gens = []
    t=0

    population = create_population(m,A)
    fit = find_fitness(population,r,A)
    # fit = find_fitness2(population, G)

    best = np.argmax(fit) #indicele celui mai fit cromozom
    cnt = 1
    old_best = fit[best] #cel mai fit cromozom atm
    gens.append((population[best,:], fit[best])) #pun cel mai bun cromozom si fit-ul lui

    while t<30 and cnt<=5:
        #rulez algoritmul pe 30 de generatii
        #sau pana cand cel mai fit cromozom nu se schimba (in 5 generatii)
        t=t+1
        sel_pop = proportional_selection(elitism, population, fit)
        cross_pop = crossover(pc, sel_pop)
        population = mutation(pm, cross_pop,A)

        fit = find_fitness(population,r,A)
        # fit = find_fitness2(population, G)

        best = np.argmax(fit)

        if fit[best] == old_best: cnt=cnt+1
        else:
            old_best = fit[best]
            cnt=1

        gens.append((population[best,:], fit[best]))

    #sortez dupa fitness function (x[1]) si returnez cel mai fit cromozom
    best_chrom = sorted(gens, key=lambda x: x[1], reverse=True)[0]
    #caut comunitatile celui mai fit cromozm
    coms = get_coms(best_chrom[0])

    return best_chrom, coms
#%%
# Import real topologies

dolphins_gml = nx.read_gml("dolphins.gml", label='id')
football_gml = nx.read_gml("football.gml", label='id')
karate_gml = nx.read_gml("karate.gml", label='id')
krebs_gml = nx.read_gml("krebs.gml", label='id')
map_edge = nx.read_edgelist("map.edge")
name_edge = nx.read_edgelist("name.edge")
got_graphml = nx.read_graphml("got.gml")
lesmis_gml = nx.read_gml("lesmis.gml")

def conv2int(G,start_value):
    nG = nx.convert_node_labels_to_integers(G, first_label=start_value)
    return nG

dolphins = conv2int(dolphins_gml,0)
football = conv2int(football_gml,0)
karate = conv2int(karate_gml,0)
krebs = conv2int(krebs_gml,0)
map = conv2int(map_edge, 0)
name = conv2int(name_edge, 0)
got = conv2int(got_graphml, 0)
lesmis = conv2int(lesmis_gml, 0)
# modularity for GA community detection algorithm
def compute_modularity(G, G_results):
    mod = nx.algorithms.community.quality.modularity(G,G_results)
    n_coms = len(G_results)
    return n_coms, mod
def run_genetic(G,m,elite,r,pc,pm):
    best_fit = 0
    best_coms = None
    for i in range(len(pc)):
        for j in range(len(pm)):
            for k in range(len(elite)):
                best, nodes = genetic_algorithm(G, m, elite[k] , r, pc[i], pm[j])

                if best[1] > best_fit:
                    best_fit = best[1]
                    best_coms = nodes

    return best_coms
#%%
# pc_values = [0.7, 0.8, 0.9]
# pm_values = [0.1,0.2]
# elitism_values = list(range(1,3))
pc_values = [0.8]
pm_values = [0.2]
elitism_values = [30]

# got_coms = run_genetic(got, 300, elitism_values, 1, pc_values, pm_values)
# got_gen_num, got_gen_mod = compute_modularity(got, got_coms)

dolphins_coms = run_genetic(dolphins, 300, elitism_values, 1, pc_values, pm_values)
# football_coms = run_genetic(football, 300, elitism_values, 0.5, pc_values, pm_values)
# krebs_coms = run_genetic(krebs, 300, elitism_values, 0.5, pc_values, pm_values)
# karate_coms = run_genetic(karate, 300, elitism_values, 0.75, pc_values, pm_values)
# map_coms = run_genetic(map, 300, elitism_values, 0.75, pc_values, pm_values)
# name_coms = run_genetic(name, 300, elitism_values, 0.75, pc_values, pm_values)

dolphins_gen_num, dolphins_gen_mod = compute_modularity(dolphins, dolphins_coms)
dolphins_Newman_num, dolphins_Newman_mod = greedyCommunitiesDetection(dolphins)

# football_gen_num, football_gen_mod = compute_modularity(football, football_coms)
# football_Newman_num, football_Newman_mod = greedyCommunitiesDetection(football)
#
# krebs_gen_num, krebs_gen_mod = compute_modularity(krebs, krebs_coms)
# krebs_Newman_num, krebs_Newman_mod = greedyCommunitiesDetection(krebs)
#
# karate_gen_num, karate_gen_mod = compute_modularity(karate, karate_coms)
# karate_Newman_num, karate_Newman_mod = greedyCommunitiesDetection(karate)
#
# map_gen_num, map_gen_mod = compute_modularity(map, map_coms)
# map_Newman_num, map_Newman_mod = greedyCommunitiesDetection(map)
#
# name_gen_num, name_gen_mod = compute_modularity(name, name_coms)
# name_Newman_num, name_Newman_mod = greedyCommunitiesDetection(name)
#
# lesmis_coms = run_genetic(lesmis, 300, elitism_values, 2, pc_values, pm_values)
# lesmis_gen_num, lesmis_gen_mod = compute_modularity(lesmis, lesmis_coms)



print("DOLPHINS")
print ("Genetic Algorithm: %r communities with modularity score %r" %(dolphins_gen_num,dolphins_gen_mod))
print("Fast Newman Algorithm: %r communities with modularity score %r" %(dolphins_Newman_num, dolphins_Newman_mod))

# print("FOOTBALL")
# print ("Genetic Algorithm: %r communities with modularity score %r" %(football_gen_num,football_gen_mod))
# print("Fast Newman Algorithm: %r communities with modularity score %r" %(football_Newman_num, football_Newman_mod))
#
# print("KREBS")
# print ("Genetic Algorithm: %r communities with modularity score %r" %(krebs_gen_num,krebs_gen_mod))
# print("Fast Newman Algorithm: %r communities with modularity score %r" %(krebs_Newman_num, krebs_Newman_mod))
#
# print("KARATE")
# print ("Genetic Algorithm: %r communities with modularity score %r" %(karate_gen_num,karate_gen_mod))
# print("Fast Newman Algorithm: %r communities with modularity score %r" %(karate_Newman_num, karate_Newman_mod))
# #%%
# print("MAP")
# print ("Genetic Algorithm: %r communities with modularity score %r" %(map_gen_num,map_gen_mod))
# print("Fast Newman Algorithm: %r communities with modularity score %r" %(map_Newman_num, map_Newman_mod))
# #%%
# print("NAME")
# print ("Genetic Algorithm: %r communities with modularity score %r" %(name_gen_num,name_gen_mod))
# print("Fast Newman Algorithm: %r communities with modularity score %r" %(name_Newman_num, name_Newman_mod))
# #%%
# print ("Genetic Algorithm: %r communities with modularity score %r" %(got_gen_num,got_gen_mod))
# print ("Genetic Algorithm: %r communities with modularity score %r" %(lesmis_gen_num,lesmis_gen_mod))