import java.util.List;

public class Instance {
	double xPoint;
	double yPoint;
    int clusterNumber;
    int instanceId;

    //Instance constructor
    public Instance(int id, double x, double y)
	{
    	xPoint = x;
    	yPoint = y;
    	instanceId = id;
    	clusterNumber  = -1;
	}

    //Getters & Setters
    public double getXPoint()
	{
		return xPoint;
	}
    public double getYPoint()
	{
		return yPoint;
	}
    public int getInstanceId()
	{
		return instanceId;
	}
    public int getClusterNumber()
	{
		return clusterNumber;
	}
	public void setXPoint(double x)
	{
		xPoint= x;
	}
	public void setYPoint(double y)
	{
		yPoint=y;
	}
	
	//assign the instance to a cluster based on less distance from centroid
	public void assignmentStep(List<Cluster> clustList)
	{
		double distance,tempDist;
		int tempCluster;
		tempDist = findDistance(clustList.get(0).centroid);
		distance = tempDist;
		tempCluster = clustList.get(0).clusterNum;
		for(int i = 1;i<clustList.size();i++)
		{
			tempDist = findDistance(clustList.get(i).centroid);
			if(tempDist < distance)
			{
				tempCluster = clustList.get(i).clusterNum;
				distance = tempDist;

			}
		}
		clusterNumber = tempCluster;
		clustList.get(clusterNumber).listOfInstances.add(this);
	}
	
	//find distance between the instance to centroid
	public double findDistance(Instance centroid)
	{
		return Math.sqrt(Math.pow((xPoint-centroid.xPoint), 2)+Math.pow((yPoint-centroid.yPoint), 2));
	}
		
}