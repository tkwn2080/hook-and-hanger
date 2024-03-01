// const { Configuration, OpenAIApi } = require("openai");
const moment = require('moment');
const {ChromaClient} = require('chromadb');
const client = new ChromaClient({
  path: "http://localhost:8000"
});

// const configuration = new Configuration({
//     apiKey: process.env.OPENAI_API_KEY,
//   });
//   const openai = new OpenAIApi(configuration);

const openai = require('openai');
openai.apiKey = process.env.OPENAI_API_KEY;
  
const {OpenAIEmbeddingFunction} = require('chromadb');
const embedder = new OpenAIEmbeddingFunction({ openai_api_key: process.env.OPENAI_API_KEY });


//the whole is structured for the exemplar which follows, a conversation between person-a and person-b
//TBD: it need not have two collections, just use metadata to structure?



//CHUNKING
//splits and chunks for at the sentence-level structured retrieval
//each chunk is associated with thread, block, and chunk numbers
async function fishChunking(text, identity) {
  try {
    text = text.toString();
    let speakerBlocks = text.split(/\n{2}(?=PERSON-[AB]:)/);
    const timestamp = moment().format();

    let ids = [];
    let documents = [];
    let metadatas = [];

    for (let i = 0; i < speakerBlocks.length; i++) {
        let block = speakerBlocks[i];
        let speakerEnd = block.indexOf(':');
        let speaker = block.slice(0, speakerEnd).trim();
        let content = block.slice(speakerEnd + 1).trim();

        content = content.replace(/\n/g, " ");
        let contentChunks = content.split(/(?<=[.!?])\s/).filter(chunk => chunk.trim().length > 0);

        contentChunks.forEach((chunk, chunkIndex) => {
          const id = `${timestamp}-${i}-${chunkIndex}`;
          ids.push(id);

            documents.push(chunk);
            
            metadatas.push({
                blockNumber: i,
                chunkNumber: chunkIndex,
                memorySource: speaker,
                memoryTime: timestamp,
            });
        });
    }


    if (identity === "Person-A") {
      await collectionA.add({
        ids: ids,
        documents: documents,
        metadatas: metadatas,
      });
    } else if (identity === "Person-B") {
      await collectionB.add({
        ids: ids,
        documents: documents,
        metadatas: metadatas,
      });
    }

    console.log(`Done!`);
  } catch (error) {
    console.log(error);
  }
}


//RETRIEVAL
//three parts: hook, hanger, filter
//the hook retrieves the lowest semantic distance chunk
//the hanger pulls up the surrounding chunks for context
//the filter trims the hanger based on a relative threshold
async function retrieveMemory(text, identity) {
  try {
    let hook = await hookQuery(text);
    let hanger = await hangerQuery(hook);
    let result = await clippingFilter(hanger);
    return result;
  } catch (error) {
    console.log(`Error in retrieveMemory: ${error}`);
  }
}


//this selects a hook, the lowest semantic distance chunk
async function hookQuery(text) {
  try {
      const results = await collectionA.query({
          queryTexts: [text],
          nResults: 7,
          include: ["metadatas", "documents", "distances"]
      });
      console.log(results);
      return results;
  } catch (error) {
      console.log(`Error in hookQuery: ${error}`);
  }
}

//this queries the area surrounding the hook, pulls up the surrounding chunks
async function hangerQuery(lowestSemDistDoc, lowestSemDistMeta, text) {
  try {
      if (!lowestSemDistDoc || !lowestSemDistMeta) {
          console.log("Invalid inputs for hangerQuery");
          return;
      }

      let blockNumber = lowestSemDistMeta.blockNumber;
      let chunkNumber = lowestSemDistMeta.chunkNumber;
      console.log(`blockNumber: ${blockNumber}, chunkNumber: ${chunkNumber}`);

      let chunkNumberMin = Math.max(0, chunkNumber - 3);
      let chunkNumberMax = chunkNumber + 3;
      console.log(`chunkNumberMin: ${chunkNumberMin}, chunkNumberMax: ${chunkNumberMax}`);

      const results2 = await collectionA.query({
          queryTexts: [text],
          nResults: 10,
          where: {
              "$and": [
                  {"blockNumber": {"$eq": blockNumber}},
                  //There is a sense that these are not limiting the search parameter, and perhaps it is not necessary. 
                  //It may be so, however, with larger messages; it is not clear and requires further testing with different sets.
                  {"chunkNumber": {"$gte": chunkNumberMin}},
                  {"chunkNumber": {"$lte": chunkNumberMax}}
              ]
          },
          include: ["metadatas", "documents", "distances"]
      });

      return results2;
  } catch (error) {
      console.log(`Error in hangerQuery: ${error}`);
  }
}

//this filters the hanger based on a relative threshold
async function clippingFilter(results, relativeThreshold = 0.3) {
  try {
      // Extract the documents, metadatas, and distances from the results
      let documents = results.documents[0];
      let metadatas = results.metadatas[0];
      let distances = results.distances[0];

      // Check if extracted arrays are of the same length
      if (documents.length !== metadatas.length || documents.length !== distances.length) {
          console.log("Mismatch in lengths of documents, metadatas, and distances arrays");
          return [];
      }

      // Zip together the documents, metadatas, and distances arrays
      let combined = documents.map((doc, i) => ({doc, metadata: metadatas[i], dist: distances[i]}));

      // Sort the combined array by chunkNumber
      combined.sort((a, b) => a.metadata.chunkNumber - b.metadata.chunkNumber);

      // Clear any duplicates (per chunkNumber)
      combined = combined.filter((x, i, a) => i === 0 || x.metadata.chunkNumber !== a[i - 1].metadata.chunkNumber);

      // Find the central point where distance is lowest
      let centralPointIndex = combined.findIndex(x => x.dist === Math.min(...combined.map(x => x.dist)));

      // Check if central point was found
      if (centralPointIndex === -1) {
          console.log("Central point not found in documents");
          return [];
      }

      // Initialize the output array with the central point
      let output = [combined[centralPointIndex]];

      // Add points to the left of the central point until the relative threshold is exceeded
      for (let i = centralPointIndex - 1; i >= 0; i--) {
          if ((combined[i + 1].dist - combined[i].dist) / combined[i + 1].dist <= relativeThreshold) {
              output.unshift(combined[i]);
          } else {
              break;
          }
      }

      // Add points to the right of the central point until the relative threshold is exceeded
      for (let i = centralPointIndex + 1; i < combined.length; i++) {
          if ((combined[i].dist - combined[i - 1].dist) / combined[i].dist <= relativeThreshold) {
              output.push(combined[i]);
          } else {
              break;
          }
      }

      // Return the output array, transformed back into separate documents, metadatas, and distances arrays
      return {
          documents: output.map(x => x.doc),
          metadatas: output.map(x => x.metadata),
          distances: output.map(x => x.dist)
      };
  } catch (error) {
      console.log(`Error in clippingFilter: ${error}`);
  }
}


//EXAMPLE
//the following code implements a test conversation between person-a and person-b
//this is intended only as a simple example of this conversational memory in action

let collectionA, collectionB;

//this initiates the collections, using chromadb with openai embeddings
async function initiateCollection() {
  try {
    // await client.deleteCollection({
    //   name: "memory-a"
    //  });

    // await client.deleteCollection({
    //   name: "memory-b"
    //  });

    try {
        collectionA = await client.getCollection({
          name: "memory-a",
          embeddingFunction: embedder,
        });
      } catch (error) {
        //do I need to specify the error?
        collectionA = await client.createCollection({
          name: "memory-a",
          metadata: {
            "description": "conversation memory for person-a"
          },
          embeddingFunction: embedder,
        });
      }

      try {
        collectionB = await client.getCollection({
          name: "memory-b",
          embeddingFunction: embedder,
        });
      } catch (error) {
        //do I need to specify the error?
        collectionA = await client.createCollection({
          name: "memory-b",
          metadata: {
            "description": "conversation memory for person-b"
          },
          embeddingFunction: embedder,
        });
      }
    
    
  } catch (error) {
    console.log(`Error in initiateCollection: ${error}`);
  }
}

//this handles the conversation between person-a and person-b
async function handleConversation(convLength) {
  try {
    let messageB = "Hello, how are you?";

    console.log (`\n\nPERSON-B\n\n${messageB}`);

    const totalMessages = convLength;
    for (let i = 0; i < totalMessages; i++) {
      if (i % 2 == 0) {
        messageA = await personA(messageB);
      } else {
        messageB = await personB(messageA);
      }
    }
  } catch (error) {
    console.log(`Error in Conversation: ${error}`);
  }
}


async function personA(otherMessage) {
  try {
    const associativeMemory = await retrieveMemory(otherMessage, "Person-A");

    memoryPrompt = await addMemory(associativeMemory, otherMessage, "collectionA", "Person-A", "Person-B");
    console.log(`\n\nMEMORY A\n\n${memoryPrompt}`);

    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      // model: "gpt-4",
      messages: [
        {role: "system", content: `I am Person-A, and I am conversing with Person-B."`},
        {role: "assistant", content: `For a moment I lose myself in an internal monologue based on memories brought to mind by this: ${memoryPrompt}`},
        {role: "user", content: `\n\nNow they have just said "${otherMessage}. With all this in mind, I begin to reply:`},
      ],
      temperature: 0.9,
    });
    completion = response.data.choices[0].message.content;
    console.log(`\n\nPERSON-A\n\n${completion}`);

    await fishChunking(`\n\nPERSON-B: ${otherMessage}\n\nPERSON-A: ${completion}`, "Person-A");

    return completion;
  } catch (error) {
    console.log(`Error in Person A: ${error.message}`);
  }
}


async function personB(otherMessage) {
  try {
    const associativeMemory = await retrieveMemory(otherMessage, "Person-B");

    memoryPrompt = await addMemory(associativeMemory, otherMessage, "collectionB", "Person-B", "Person-A");
    console.log(`\n\nMEMORY B\n\n${memoryPrompt}`);

    response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      // model: "gpt-4",
      messages: [
        {role: "system", content: `I am Person-B, and I am conversing with Person-A."`},
        {role: "assistant", content: `For a moment I lose myself in an internal monologue based on memories brought to mind by this: ${memoryPrompt}`},
        {role: "user", content: `\n\nNow they have just said "${otherMessage}. With all this in mind, I begin to reply:`},
      ],
      temperature: 0.9,
    });
    completion = response.data.choices[0].message.content;
    console.log(`\n\nPERSON-B\n\n${completion}`);

    await fishChunking(`\n\nPERSON-A: ${otherMessage}\n\nPERSON-B: ${completion}`, "Person-B");

    return completion;
  } catch (error) {
    console.log(`Error in Person B: ${error.message}`);
  }
}

//this integrates meories by filtering them through a monologue
async function addMemory(associativeMemory, otherMessage, collectionName, identity1, identity2) {
  try {
    let memoryPrompt = "";

    if (associativeMemory.ids[0].length > 0) {
      memoryPrompt = `Your conversation partner has just said: ${otherMessage}\n\nThis has made you remember the following:\n`;

      associativeMemory.documents[0].forEach((doc, index) => {
        let distance = associativeMemory.distances[0][index];
        let id = associativeMemory.ids[0][index];
        memoryPrompt += `Memory date and time ${id}:\n"${doc}"\n`;
      });

      const response = await openai.chat.completions.create({
        model: "gpt-3.5-turbo",
        // model: "gpt-4",
        messages: [
          {
            role: "system",
            content: `I am the inner monologue and memory of ${identity1}. I am having a conversation with another human, ${identity2}. Here I take a set of associative memories that have come to mind and filter them for relevance, then present it to myself as an internal monologue. I should note the semantic and temporal distances of each memory, that these are important for understanding them. The current time is ${moment().format("dddd, MMMM Do YYYY, h:mm:ss a")}.`
          },
          {role: "user", content: `${memoryPrompt}\n\nI will now consider these memories and think through their implications for my behaviour. I will limit my thoughts to the memories that have been presented here. My monologue will have the following structure: memory one ('I recall that'), implications; memory two ('I recall that'), implications; finally synthesising these considerations.`},
        ],
        temperature: 0.7,
      });
      const completion = response.data.choices[0].message.content;
      memoryPrompt = completion;
      return memoryPrompt;
    }
  } catch (error) {
    console.log(`Error in addMemory: ${error}`);
  }
}

function sleep(minutes) {
  let ms = minutes * 60 * 1000;
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function testConversation(length) {
  try {
    await initiateCollection();

    for (let i = 0; i < length; i++) {
      await handleConversation(5);
      console.log(`Sleeping for 2 minutes...`);
      await sleep(2);
    }

  } catch (error) {
    console.log(`Error in test: ${error}`);
  }
}

// testConversation(2);


async function testQuery(query) {
  try {
    console.log(query);
    let hook = await hookQuery(query);
    console.log(hook);
    let hanger = await hangerQuery(hook);
    console.log(hanger);
    let results = await clippingFilter(hanger);
    console.log(results);
  } catch (error) {
    console.log(`Error in testQuery: ${error}`);
  }
}

testQuery("I am happy")