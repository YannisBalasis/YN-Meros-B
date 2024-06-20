import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import sys
import codecs
import re
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

# δοσμένη επιγραφή
epigrafi = "[...] αλεξανδρε ουδις [...]"

# Λεξικό μεγέθους 1678 tokens
lexicon_size = 1678


dataset = pd.read_csv("C:/Python/Computational Intelligence/Exercise1/iphi2802.csv", header=0, sep='\t', encoding='utf-8')

epigrafi_data = dataset["text"].tolist()

#  tokenizer
def tokenize_text(text):
    words = word_tokenize(text)  
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words] 
    words = [word for word in filtered_words if re.match(r'\b\w+\b', word)]  
    return words

# Προεπεξεργασία κειμένου
preprocessed_text = [" ".join(tokenize_text(text.lower())) for text in epigrafi_data]
# Προσθήκη της δοσμένης επιγραφής στη λίστα των επιγραφών
preprocessed_text.append(" ".join(tokenize_text(epigrafi.lower())))
# Έλεγχος για διπλές τιμές 
unique_preprocessed_text = list(set(preprocessed_text))


tfidf_vectorizer = TfidfVectorizer(max_features=lexicon_size)

# Εκπαίδευση tf-idf vectorizer στα προεπεξεργασμένα δεδομένα επιγραφής
tfidf_vectorizer.fit(unique_preprocessed_text)

# Μετατροπή της επιγραφής σε BoW representation με χρήση tf-idf
tfidf_matrix = tfidf_vectorizer.transform(unique_preprocessed_text)
lexicon = tfidf_vectorizer.get_feature_names_out()
lexicon_list = list(lexicon)

# Έλεγχος για διπλές τιμές στο λεξικό
unique_lexicon_list = list(set(lexicon_list))

# Δημιουργία πληθυσμού
def generate_population(population_size, lexicon_list):
    population = set()
    while len(population) < population_size:
        individual = tuple(random.sample(lexicon_list, 2))  # Επιλογή τυχαίων λέξεων από το λεξικό
        population.add(individual) 
    return list(population)

population_size = 1000  # Μέγεθος  πληθυσμού
population = generate_population(population_size, unique_lexicon_list)  # Δημιουργία 

# Εκτύπωση των features (tokens) του BoW representation
print("Features (tokens) του BoW representation:\n", tfidf_vectorizer.get_feature_names_out())
# Εκτύπωση του BoW representation της επιγραφής
print("\nBoW representation της επιγραφής:\n", tfidf_matrix[-1].toarray())

# Υπολογισμός ομοιότητας συνημιτόνου
cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

# Εύρεση των top-5 ή top-10 πιο κοντινών επιγραφών
top_n = 5
similarities = cosine_similarities[0]
top_indices = similarities.argsort()[-top_n:][::-1]

print(f"Top-{top_n} πιο κοντινές επιγραφές:")
top_similarities = []
for index in top_indices:
    print(f"Επιγραφή: {epigrafi_data[index]}, Ομοιότητα: {similarities[index]}")
    top_similarities.append(similarities[index])

def fitness_function(individual, top_similarities):
    #  αντικατάσταση του "[...]"
    filled_text = epigrafi.replace("[...]", " ".join(individual))
    
    # Tokenize στο νέο κείμενο 
    preprocessed_filled_epigrafi = " ".join(tokenize_text(filled_text.lower()))
    
    # BoW representation με tf-idf
    filled_epigrafi_bow = tfidf_vectorizer.transform([preprocessed_filled_epigrafi])
    
    # Υπολογισμός ομοιότητας του κειμένου με τα τοπ-5
    filled_epigrafi_similarities = cosine_similarity(filled_epigrafi_bow, tfidf_matrix[top_indices])
    
    # επιστροφή μέσου όρου 
    fitness = filled_epigrafi_similarities.mean()
    
    return fitness

# Υπολογισμός καταλληλότητας για κάθε άτομο του πληθυσμού
fitness_values = [fitness_function(individual, top_similarities) for individual in population]

# Εύρος τιμών συνάρτησης καταλληλότητας
max_fitness = max(fitness_values)
min_fitness = min(fitness_values)

print(f"Μέγιστη τιμή συνάρτησης καταλληλότητας: {max_fitness}")
print(f"Ελάχιστη τιμή συνάρτησης καταλληλότητας: {min_fitness}")

def tournament_selection(population, fitness_values, k=3):
    selected_individuals = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness_values)), k)
        winner = max(tournament, key=lambda x: x[1])
        selected_individuals.append(winner[0])
    return selected_individuals

def multi_point_crossover(parent1, parent2, num_points=2):
    num_points = min(num_points, len(parent1) - 1, len(parent2) - 1)
    if num_points < 1:
        return parent1, parent2
    
    max_index = min(len(parent1), len(parent2))
    if max_index <= num_points:
        num_points = max_index - 1
    
    points = sorted(random.sample(range(1, max_index), num_points))
    
    offspring1 = list(parent1)
    offspring2 = list(parent2)
    
    for i in range(0, num_points, 2):
        start = points[i]
        end = points[i+1] if i+1 < num_points else max_index
        offspring1[start:end], offspring2[start:end] = offspring2[start:end], offspring1[start:end]
    
    return tuple(offspring1), tuple(offspring2)

def elitism(population, fitness_values, elite_size=2):
    elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:elite_size]
    elite_individuals = [population[i] for i in elite_indices]
    return elite_individuals

def mutation(individual, lexicon_list, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutate_index = random.randint(0, len(individual) - 1)
        new_word = random.choice(lexicon_list)
        individual = individual[:mutate_index] + (new_word,) + individual[mutate_index+1:]
    return individual

def run_genetic_algorithm(population_size, crossover_prob, mutation_prob):
    num_generations = 1000  #  αριθμός γενεών
    elite_size = 2
    mutation_rate = mutation_prob
    
    # Αρχικοποίηση πληθυσμού
    population = generate_population(population_size, unique_lexicon_list)
    
    max_fitness_history = []  # καλύτερη καταλληλότητα κάθε γενεάς
    generation_count = 0
    no_improvement_count = 0
    
    while generation_count < num_generations:
        # Υπολογισμός καταλληλότητας
        fitness_values = [fitness_function(individual, top_similarities) for individual in population]
        
        # Ενημέρωση  καλύτερης καταλληλότητας
        max_fitness_history.append(max(fitness_values))
        
        # Έλεγχος αν η λίστα max_fitness_history έχει τουλάχιστον δύο στοιχεία
        if len(max_fitness_history) > 1:
            # Έλεγχος για τερματισμό αν η βελτίωση είναι κάτω από ένα ποσοστό
            if max_fitness_history[-1] - max_fitness_history[-2] < 0.01 * max_fitness_history[-1]:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
        
        # Έλεγχος για τερματισμό αν έχει ξεπεραστεί ο προκαθορισμένος αριθμός γενεών
        if no_improvement_count == 50 or generation_count == num_generations - 1:
            break
        
        # Ελιτισμός
        new_population = elitism(population, fitness_values, elite_size)
        
        # Επιλογή
        selected_individuals = tournament_selection(population, fitness_values)
        
        # Διασταύρωση
        for i in range(0, len(selected_individuals), 2):
            if i+1 < len(selected_individuals):
                offspring1, offspring2 = multi_point_crossover(selected_individuals[i], selected_individuals[i+1])
                new_population.extend([offspring1, offspring2])
        
        # Μετάλλαξη
        new_population = [mutation(individual, unique_lexicon_list, mutation_rate) for individual in new_population]
        
        # Ενημέρωση πληθυσμού
        population = new_population[:population_size]
        
        # Ενημέρωση μετρητή γενεών
        generation_count += 1
        
        # Εκτύπωση της προόδου κάθε 100 γενεών
        if generation_count % 100 == 0:
            print(f"Γενιά {generation_count}: Μέγιστη καταλληλότητα = {max(fitness_values)}, Μέση καταλληλότητα = {sum(fitness_values) / len(fitness_values)}")
    
    # Τελικός πληθυσμός
    fitness_values = [fitness_function(individual, top_similarities) for individual in population]
    best_individual = population[fitness_values.index(max(fitness_values))]
    best_fitness = max(fitness_values)
    
    # Επιστροφή των αποτελεσμάτων
    return max_fitness_history, generation_count, best_individual, best_fitness


# Παράμετροι πίνακα
parameter_table = [
    (20, 0.6, 0.00),
    (20, 0.6, 0.01),
    (20, 0.6, 0.10),
    (20, 0.9, 0.01),
    (20, 0.1, 0.01),
    (200, 0.6, 0.00),
    (200, 0.6, 0.01),
    (200, 0.6, 0.10),
    (200, 0.9, 0.01),
    (200, 0.1, 0.01)
]
# Αποθήκευση των καμπυλών εξέλιξης
evolution_curves = []
for params in parameter_table:
    pop_size, crossover_prob, mutation_prob = params
    max_fitness_history, num_generations_used, best_individual, best_fitness = run_genetic_algorithm(pop_size, crossover_prob, mutation_prob)
    print(f"\nΠαράμετροι: Μέγεθος πληθυσμού = {pop_size}, Πιθανότητα διασταύρωσης = {crossover_prob}, Πιθανότητα μετάλλαξης = {mutation_prob}")
    print(f"Μέση τιμή μέγιστου κατά τη διάρκεια των γενεών: {sum(max_fitness_history) / len(max_fitness_history)}")
    print(f"Μέσος αριθμός γενεών που απαιτήθηκαν: {num_generations_used}")
    print(f"Καλύτερη λύση: {best_individual}, Καταλληλότητα: {best_fitness}")
    evolution_curves.append((max_fitness_history, params, best_individual))


# Σχεδίαση των καμπυλών εξέλιξης
plt.figure(figsize=(12, 8))

for i, (max_fitness_history, params, best_individual) in enumerate(evolution_curves):
    plt.plot(max_fitness_history, label=f'Population={params[0]}, Crossover={params[1]}, Mutation={params[2]}')

plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.title('Evolution of the Best Solution Over Generations')
plt.legend()
plt.show()

