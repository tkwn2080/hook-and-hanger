# Hook-and-hanger, structured queries for RAG
Embeddings-based retrieval is excellent but imperfect
This project aims to more closely resemble human memory
We recall an item based on similarity (as with cosine similarity to a chunk)
The difference is that we can branch out to this point, before and afterwards

### Balancing competing interests
Sentence-level chunking may be too specific, misses context
Larger units will dilute the cosine similarity measurements

### Structured embedding
Here we maintain the larger-scale chunk structure in metadata
Chunks are embedding with this structure, which is then used during retrieval

### Three-stage retrieval: hook, hanger, filter
The initial retrieval is a 'hook'--finds the lowest semantic distance
The secondary retrieval is a 'hanger'--branches out around this by n chunks 
The third step is a filter--selects a contiguous section around the hook
This filter operates on relative semantic distance between neighbours
The cut occurs where there is a drop-off above a certain threshold (default = 0.3)

### Further additions
We can use GPT-4 to design queries, either in terms of wording or targeted in terms of metadata
There is a recency function, which will re-order the initial hook query to emphasise recent additions
The retrieved memories are integrated into the conversation by way of an inner monologue function

# To-do
1. Multiple hook queries, multiple hangers
2. Polish recency function, requires testing
3. Allow for variation in terms of hanger size, etc.
4. Increase the example complexity, "time apart"
