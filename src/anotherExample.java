import weka.core.Instances;

import java.awt.*;
import java.io.*;
import java.util.*;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Hashtable;
import java.util.List;
import java.util.Random;
import javax.swing.*;

import weka.classifiers.functions.LinearRegression;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.classifiers.timeseries.WekaForecaster;
import weka.core.Utils;

/**
 * Created by cycle on 10.11.16.
 */
public class anotherExample extends Thread{
    private Thread t;
    private String threadName;

    anotherExample (String name) {
        threadName = name;
        System.out.println("Creating " + threadName);
    }
    public void run() {
        System.out.println("Running " +  threadName );
        try {
            for(int i = 4; i > 0; i--) {
                System.out.println("Thread: " + threadName + ", " + i);
                // Let the thread sleep for a while.
                Thread.sleep(50);
            }
        }catch (InterruptedException e) {
            System.out.println("Thread " +  threadName + " interrupted.");
        }
        System.out.println("Thread " +  threadName + " exiting.");
    }

    public void start () {
        System.out.println("Starting " +  threadName );
        if (t == null) {
            t = new Thread (this, threadName);
            t.start ();
        }
    }
    public static void main(String[] args) {
       anotherExample T1 = new anotherExample("Thread-1");
       T1.start();
    }
}
