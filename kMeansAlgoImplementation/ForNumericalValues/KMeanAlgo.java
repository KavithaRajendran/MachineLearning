import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.io.BufferedReader;
import java.io.FileReader;

public class KMeanAlgo{	
	static List<Instance> xyList = new ArrayList();
	static List<Cluster> clusterList = new ArrayList();
	
	public static void main(String[] args) throws IOException, FileNotFoundException {
		String inputFile = args[1];
		String outputFile = args[2];
        
		readPoints(inputFile);

		int k;
		k = Integer.parseInt(args[0]);
		int m = 0;
		while(m < k) {
			Cluster c = new Cluster(m);
			c.initCentroid(xyList);
			clusterList.add(c);
			m++;
		}
	
		KMeansAlgorithm(k);
		double error = calculateError();
		PrintWriter writer;
		try {
			writer = new PrintWriter(new File(outputFile));
			for(int i = 0;i<k;i++)
			{
				Cluster outputCluster  = clusterList.get(i);
				int clusterNum =outputCluster.clusterNum+1; 
				writer.println(clusterNum);
				for(int j = 0;j<outputCluster.listOfInstances.size();j++)
				{
					writer.print(outputCluster.listOfInstances.get(j).getInstanceId()+ ",");
				}
				writer.println();
			}
			writer.println("\n Sum of Squared Error = "+error);	
			writer.close();
		} catch (FileNotFoundException e1) {
			e1.printStackTrace();
		}
	}
	
	//calculate Sum of Squared Error
	public static double calculateError()
	{
		double error = 0;
		for(int i = 0; i<clusterList.size();i++)
		{
			Cluster temp = clusterList.get(i);
			for(int j = 0;j<clusterList.get(i).listOfInstances.size();j++)
			{
				error += Math.pow(temp.listOfInstances.get(i).findDistance(temp.centroid),2); 
			}
		}
		return error;
	}

	//K Means Algorithm implementation
	public static void KMeansAlgorithm(int k)
	{
		int count = 0;
		int change = 1;
		int itrCount =0;
		
		while(count<25 && change==1)
		{	
			//remove previous tweets from the cluster
			for(int i =0; i<k;i++)
			{
				if(clusterList.get(i).listOfInstances!= null)
					clusterList.get(i).listOfInstances.clear();
			}
			//reassignment tweets to new clusters
			for(int j = 0; j< xyList.size();j++)
			{
				xyList.get(j).assignmentStep(clusterList);
			}
			
			//find new centroids
			itrCount = 0;
			for(int i = 0;i<k;i++)
			{
				if(false == clusterList.get(i).findCentroid())
				{
					itrCount++;
				}
			}
			//iteration termination condition
			if(itrCount == 25)
			{
				change = 0;
			}
			count++;
		}
	}
	//Read tweets from input file
	public static void readPoints(String inputFile) throws IOException
	{
		
		BufferedReader bf = new BufferedReader(new FileReader(inputFile));
		String line = "";
		String[] spiltted;
		Instance t;
		int count = 1;
		while((line = bf.readLine()) != null) {
			spiltted = line.split("\t");
			if(count == 1) {
				count++;
				continue;
			}
			t = new Instance(Integer.parseInt(spiltted[0]),Double.parseDouble(spiltted[1]),Double.parseDouble(spiltted[2]));
			xyList.add(t); //Storing input data entry as an object in a list
		}
		bf.close();
	}
}
