/* 
* Description: Cluster centroid initialization, re-assigment is done
* Author: Kavitha Rajendran */
import java.util.List;
import java.util.ArrayList;
import java.util.Random;

public class Cluster {
	int clusterNum;
	Instance centroid;
	List<Instance> listOfInstances = new ArrayList();
	
    //Cluster constructor
	public Cluster(int no)
	{
		clusterNum = no;
	}

	public void initCentroid(List<Instance> dList)
	{
		int index = 0,flag = 1;
		Random rand = new Random();
		List<Integer>temp = new ArrayList();
		while(flag == 1)
		{
		index = rand.nextInt(dList.size());
		if(false == temp.contains(index))
			flag = 0;
		}
		centroid = new Instance(-1,-1,-1);
		centroid.setXPoint(dList.get(index).getXPoint());
		centroid.setYPoint(dList.get(index).getYPoint());
	}
	
	public boolean findCentroid()
	{
		Instance tempcentroid = setCentroid();
		if(tempcentroid.getXPoint() == centroid.getXPoint() && tempcentroid.getYPoint() == centroid.getYPoint())
			return false;
		else
		{
			centroid.setXPoint(tempcentroid.getXPoint());
			centroid.setYPoint(tempcentroid.getYPoint());
			return true;
		}
	}
	Instance setCentroid()
	{
		int k;
		Instance tempCentroidPoint = new Instance(-1,-1,-1);
		double totalX =0,totalY =0;
		for(k=0;k<listOfInstances.size();k++)
		{
			totalY += listOfInstances.get(k).getYPoint();
			totalX += listOfInstances.get(k).getXPoint();
		}
		tempCentroidPoint.setYPoint(totalY/(double) listOfInstances.size());
		tempCentroidPoint.setXPoint(totalX/(double) listOfInstances.size());
		return tempCentroidPoint;
	}
}
