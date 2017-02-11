package src;

import weka.core.Instances;

/**
 * Created by cycle on 11.02.2017.
 */
public class MyThread extends Thread{
    private  String threadName;
    public MyHashMap hashMap;
    private Instances data;
    private int startLag, endLag, lagInterval, lagLimit;
    private int featureNumber;

    MyThread(String threadName, MyHashMap hashMap, Instances data, int startLag, int endLag, int featureNumber, int lagLimit, int lagInterval){
        this.threadName = threadName;
        this.hashMap = hashMap;
        this.data = data;
        this.startLag = startLag;
        this.endLag = endLag;
        this.featureNumber = featureNumber;
        this.lagLimit = lagLimit;
        this.lagInterval = lagInterval;
    }
    public void run() {
        System.out.println("Running " +  threadName );
        boolean breakLoop = false;
        for (int i = startLag; i < lagLimit; i += lagInterval) {
            if(i+lagInterval-1 > lagLimit){
                endLag = lagLimit;
                breakLoop = true;                                   //to break after ranking the last interval
            }else
                endLag = i+lagInterval-1;
            String result = applyFilterClassifier.applyFilterClassifier(data, i, endLag);
            addToHashmap(result);
            if(breakLoop)
                break;
        }
        System.out.println("Thread " +  threadName + " exiting.");
    }
    private synchronized void addToHashmap(String result){
        hashMap.fillUpHashMap(result, featureNumber, data.attribute(1).name());
    }

    public int getStartLag() {
        return startLag;
    }

    public void setStartLag(int startLag) {
        this.startLag = startLag;
    }

    public int getEndLag() {
        return endLag;
    }

    public void setEndLag(int endLag) {
        this.endLag = endLag;
    }

    public int getFeatureNumber() {
        return featureNumber;
    }

    public void setFeatureNumber(int featureNumber) {
        this.featureNumber = featureNumber;
    }
}
