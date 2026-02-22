from transformers import AutoTokenizer 

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/msmarco-bert-base-dot-v5")

def create_sentences(file_path) :
    sentences = []
    try :
        with open(file_path) as file :
                for sent in file :
                    sentences.append(sent)
    except FileNotFoundError :
        print("erreur")
    return sentences
        
        
class Chunks_creator :
    
    def __init__(self,sentences):
        self.sentences = sentences

    def create_chunks(self,max_tokens = 60,overlap = 10):
        chunks = []
        last_overlap = ''
        for sen in self.sentences:
            
            if last_overlap :
                sen = last_overlap + " " + sen
                
                
            sen_tokens = tokenizer.encode(sen,add_special_tokens=False)    
            
            if len(sen_tokens) > max_tokens :
                start = 0
                while start < len(sen_tokens) :
                    end = min(start + max_tokens,len(sen_tokens))
                    new_sen = sen_tokens[start:end]
                    new_text = tokenizer.decode(new_sen,skip_special_tokens=True)
                    chunks.append(new_text)
                    start += max_tokens - overlap
                    
                overlap_tokens = new_sen[-overlap:]
                last_overlap = tokenizer.decode(overlap_tokens,skip_special_tokens = True)
                
            else :
                chunks.append(sen)
                overlap_tokens = sen_tokens[-overlap:]
                last_overlap = tokenizer.decode(overlap_tokens,skip_special_tokens = True)
    
            
        return chunks
 
        

