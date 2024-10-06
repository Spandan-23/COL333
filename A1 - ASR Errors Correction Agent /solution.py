class Agent(object):
    def __init__(self, phoneme_table, vocabulary) -> None:
        self.phoneme_table = phoneme_table
        self.vocabulary = vocabulary
        self.best_state = None
        self.get_phoneme()

    def get_phoneme(self):
        new_phoneme={}
        for key in self.phoneme_table.keys():
            for value in self.phoneme_table[key]:
                if value not in new_phoneme:
                    new_phoneme[value] = [key]
                else:
                    new_phoneme[value].append(key)
        self.phoneme_table=new_phoneme

    def get_neighbours(self, curr_state,curr_cost,environment):
        #print(f"\nGenerating phoneme neighbours for: '{curr_state}'")
        best_found_state=curr_state
        # best_cost = curr_cost
        for i in range(len(curr_state)):
            char = ""
            for j in range(min(2, len(curr_state) - i)):
                char += curr_state[i + j]
                if char in self.phoneme_table:
                    for subs in self.phoneme_table[char]:
                        s = curr_state[:i] + subs + curr_state[(i + j + 1):]
                        cost = environment.compute_cost(s)
                        if cost<=curr_cost:
                            best_found_state=s
                            curr_cost=cost

        return best_found_state,curr_cost
    def get_neighbours2(self, curr_state,curr_cost,environment,iter):
        best_found_state=curr_state
        #print(f"\nGenerating vocabulary addition neighbours for: '{curr_state}'")
        if iter==0:
            for i in self.vocabulary:
                n = i+" "+curr_state
                cost = environment.compute_cost(n)
                #print(n,cost)
                if cost<=curr_cost:
                    curr_cost=cost
                    best_found_state=n
        else:
            for i in self.vocabulary:
                n = curr_state+" "+i
                cost = environment.compute_cost(n)
                #print(n,cost)
                if cost<=curr_cost:
                    curr_cost=cost
                    best_found_state=n

        return best_found_state,curr_cost
    def get_vocab_words(self,reverse_vocab_state,initial_state):
        words=["",""]
        L1=reverse_vocab_state.split(' ')
        L2=initial_state.split(' ')
        if len(L1)-len(L2) ==2:
            words[0]=L1[0]+" "
            words[1]=" "+L1[-1]
        elif len(L1)-len(L2)==1:
            if L1[0]!=L2[0] :
                words[0]=L1[0]+" "    
            else:
                words[1]=" "+L1[-1]
        return words

    def asr_corrector(self,environment):
        self.best_state=environment.init_state
        reverse_best_state = environment.init_state
        curr_cost = environment.compute_cost(self.best_state)
        reverse_cost = curr_cost

        for iter in range(2):
            #print(f"ITERATION {iter}")
            #print("Current State : ",reverse_best_state, " Cost : ",reverse_cost)
            best_neighbour,reverse_cost = self.get_neighbours2(reverse_best_state,reverse_cost,environment,iter)
            # if reverse_best_state!=best_neighbour:
                #print("Best reverse vocab neighbour found :  ",best_neighbour, " Cost : ",reverse_cost)
            reverse_best_state=best_neighbour
        reverse_words = self.get_vocab_words(reverse_best_state,self.best_state)
        #print(reverse_words)
        #print("Best reverse vocab state found : ",reverse_best_state, " Cost: ",reverse_cost)
        #print("------------------------------------------------------------------------------")
        #print("------------------------------------------------------------------------------")
        MAX_ITER=1000
        for iter in range(MAX_ITER):
            #print(f"ITERATION {iter}")
            #print("Current State : ",self.best_state, " Cost : ",curr_cost)
            best_neighbour,curr_cost = self.get_neighbours(self.best_state,curr_cost,environment)
            if self.best_state==best_neighbour:
                break
            #print("Best phoneme neighbour found :  ",best_neighbour, " Cost : ",curr_cost)
            self.best_state=best_neighbour
            # curr_cost=best_cost
        #print("Best phoneme state found : ",self.best_state, " Cost: ",curr_cost)
        #print("-----------------------------------------------------------------------------")
        reverse_best_state = self.best_state
        for iter in range(2):
            #print(f"ITERATION {iter}")
            #print("Current State : ",self.best_state, " Cost : ",curr_cost)
            best_neighbour,curr_cost = self.get_neighbours2(self.best_state,curr_cost,environment,iter)
            # if self.best_state!=best_neighbour:
                #print("Best phoneme neighbour found :  ",best_neighbour, " Cost : ",curr_cost)
            self.best_state=best_neighbour
            # curr_cost=best_cost
        reverse_best_state = reverse_words[0]+reverse_best_state+reverse_words[1]
        reverse_cost = environment.compute_cost(reverse_best_state)
        if reverse_cost < curr_cost:
            self.best_state = reverse_best_state
            curr_cost = reverse_cost
        #print("Best phoneme state found : ",self.best_state, " Cost: ",curr_cost)
