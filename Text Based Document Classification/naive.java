import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.File;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;

public class naive {
	static int unseen;
	static HashMap<String, Integer> vocabulary = new HashMap<>();
	static List<String> classNames= new ArrayList<String>();
	static int documentCount;
	static HashMap<String, Integer> classDocCount = new HashMap<>();
	static HashMap<String, Double> prior = new HashMap<>();
	static HashMap<String, HashMap<String, Double>> contionalProb = new HashMap<>();
	static HashMap<String, Double> unSeenWordProb = new HashMap<>();
	static HashMap<String, Integer> stopWords = new HashMap<>();
	static int correctlyPredicted;
	static int testTotalDocs;
	
	public static void trainMNB(File trainFilePath, String stopWordsString) throws IOException {
		
		//Building HashMap of stop words
		String[] sw = stopWordsString.split(",");
		for(String s:sw) {
			stopWords.put(s, 1);
		}

		//Building vocabulary
		File[] listOfClasses = trainFilePath.listFiles();	    //Listing files in a directory
		if(listOfClasses != null) {
			//Building Vocabulary
			for(File c: listOfClasses) {
				int documentCountInAClass=0;
				//System.out.println(c.getName());
				classNames.add(c.getName());
				File[] filesWithInClass = c.listFiles();
				//For all files in each class
				for(File f : filesWithInClass) {
					documentCount++; 					//COUNTDOCS(D)
					documentCountInAClass++; 			//COUNTDOCSINCLASS
					extractWords(f,vocabulary);					//Build Vocabulary
				}
				classDocCount.put(c.getName(),documentCountInAClass);
				//System.out.println("class name:"+c.getName());
				System.out.println("Number of documents in training data: "+documentCountInAClass+" for class: "+c.getName());
				//System.out.println(classDocCount);
			}
			System.out.println("Total Number of documents N="+documentCount);
			
			//Calculating prior
			calculatePrior();
			
			//Calculating conditional prob
			for(File c: listOfClasses) {
				HashMap<String, Integer> textc = concatenateTextOfAllDocsInClass(c);
				int totalWordsWithDup =0; 
				for(String key: vocabulary.keySet()) {
					if(textc.containsKey(key))
						totalWordsWithDup+=textc.get(key)+1;
					else
						totalWordsWithDup+=1;
				}
				HashMap<String, Double> innerHashForThisClass = new HashMap<>();
				//int denominator = vocabulary.size()+totalWordsWithDup;
				calculateConditionalProb(c,textc,totalWordsWithDup,innerHashForThisClass);
				contionalProb.put(c.getName(), innerHashForThisClass);
			}
		}
	}
	
	//EXTRACTVOCABULARY(D)
	public static void extractWords(File inputFile, HashMap<String, Integer> map) {
		try{
		    BufferedReader reader = new BufferedReader(new FileReader(inputFile));
		    String line;
		    while ((line = reader.readLine()) != null)
		    {
		    	//System.out.println(line);
		    	for(String word: line.split(" ")) {
		    		//System.out.println(word);
			    	//if not stop word
			    	//word = word.replaceAll("[^a-zA-Z]", "").toLowerCase().trim();
			   		word = word.toLowerCase().trim();
			   		if(word.length()>0) {
			   			if(!stopWords.containsKey(word)){
			    			//System.out.println(word);
			    			if(map.containsKey(word))
			    				map.put(word, map.get(word)+1);
			    			else
			   					map.put(word, 1);
			   			}
			   		}
			   	}
		    }
		    reader.close();
		  } catch (Exception e){
			    System.err.format("Exception occurred trying to read '%s'.", inputFile);
			    e.printStackTrace();
		  }
	}
		
	//prior[c] = Nc/N
	public static void calculatePrior() {
		//System.out.println("Total Number of documents:"+documentCount);
		for(String key: classDocCount.keySet()) {
			int Nc = classDocCount.get(key);				
			prior.put(key, (double)Nc/documentCount);
		}
	}

	//calculating number of words in all input documents of  a class
	public static HashMap<String, Integer> concatenateTextOfAllDocsInClass(File classFolder) throws IOException {
		File[] docsInClass = classFolder.listFiles();
		HashMap<String, Integer> textc = new HashMap<>();
		for(File docs : docsInClass) {
			extractWords(docs,textc);
		}
		return textc;
	}
	
	//conditional probability calculation of a class for list of all words in vocabulary
	public static void calculateConditionalProb(File classFolder, HashMap<String, Integer> textc,int denominator,HashMap<String, Double> innerHashForThisClass) throws IOException{
		//System.out.println("calculating ConditionalProb for class :"+classFolder.getName());
		//for every word in vocabulary
		int numerator = 0;
		for(String word: vocabulary.keySet()) {
			numerator = countTokensOfTerms(textc,word)+1;
			double condProbValue = ((double)numerator)/((double)denominator);
			innerHashForThisClass.put(word, condProbValue);
		}
		double unseenProb = (double)(1.0/denominator);
		unSeenWordProb.put(classFolder.getName(), unseenProb);
	}
	
	//Get count of a given word in training data of a class
	public static int countTokensOfTerms(HashMap<String, Integer> textc, String word){
		int tct=0;
		if(textc.containsKey(word)) {
			tct = textc.get(word);
		}
		return tct;
	}
	//Predict values using MultiNomial Naive Bayse
	public static void applyMultiNomialNB(File file) {
		File[] listOfClasses = file.listFiles();
		if(listOfClasses != null) {
			//Building Vocabulary
			for(File c: listOfClasses) {
				File[] doc = c.listFiles();
				for(File d : doc) {
					testTotalDocs++;
					//System.out.println(d.getName());
					String predictedClass = applyMultinomialTest(d);
					String targetClass = c.getName().toString();
					if(!(predictedClass==null)) {
						if(predictedClass.equals(targetClass)) {
							//System.out.println("correctly predicted:"+predictedClass);
							correctlyPredicted++;
						}
					}
				}
				System.out.println("total tests document in class: "+c.getName()+" is: "+doc.length);
				//System.out.println("correctlyPredicted"+correctlyPredicted);
			}
		}
	}

	//Predict values for a given test file
	public static String applyMultinomialTest(File d) {
		String predictedClass = null;
		int i;
		double scoretemp = Double.NEGATIVE_INFINITY;
		HashMap<String, Double> score = new HashMap<>();
		for(i=0;i<classNames.size();i++) {
			String c = classNames.get(i);
			double p=prior.get(c);
			scoretemp = Math.log(p);
			HashMap<String, Integer> W = new HashMap<>();
			extractWords(d,W);
			for(String w:W.keySet()) {
				HashMap<String, Double> temp = new HashMap<>();
				temp = contionalProb.get(c);
				
				if(!temp.containsKey(w)) {
					unseen++;
					double cp = unSeenWordProb.get(c);
					scoretemp+=Math.log(cp);
				}
				else {
					double cp = temp.get(w);
					scoretemp+=Math.log(cp);
				}
			}
			score.put(c, scoretemp);
			//System.out.println(" ");
		}
		
		double highestVote = Double.NEGATIVE_INFINITY;
	    for (HashMap.Entry<String, Double> temp : score.entrySet()) {
	        if (temp.getValue() > highestVote) {
	        	highestVote = temp.getValue();
	        	predictedClass = temp.getKey();
		    }
        }
		return predictedClass;
	}
	
	public static void main(String args[]) throws IOException {
		//command line args 
		File trainFilePath = new File(args[0]);
		File testFilePath = new File(args[1]);
		//File stopWordsFilePath = new File(args[2]);
		String stopWords = "a,able,about,across,after,all,almost,also,am,among,an,and,any,are,as,at,be,because,been,but,by,can,cannot,could,dear,did,do,does,either,else,ever,every,for,from,get,got,had,has,have,he,her,hers,him,his,how,however,i,if,in,into,is,it,its,just,least,let,like,likely,may,me,might,most,must,my,neither,no,nor,not,of,off,often,on,only,or,other,our,own,rather,said,say,says,she,should,since,so,some,than,that,the,their,them,then,there,these,they,this,tis,to,too,twas,us,wants,was,we,were,what,when,where,which,while,who,whom,why,will,with,would,yet,you,your";
		//Train Naive Bayse
		trainMNB(trainFilePath,stopWords);
		
		//Predict
		applyMultiNomialNB(testFilePath);
		
		//Accuracy calculation
		System.out.println("correctlyPredicted:"+correctlyPredicted);
		System.out.println("documentCount:"+testTotalDocs);
		double acc = (double)correctlyPredicted/testTotalDocs;
		System.out.println("Accuracy:"+(acc*100));
	}
}
