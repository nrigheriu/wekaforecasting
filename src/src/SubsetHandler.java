package src;

import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Random;

/**
 * Created by cycle on 25.02.2017.
 */
public class SubsetHandler {
    public int m_numAttribs;
    /**
     * Attributes which should always be included in the subset list
     */
    public ArrayList<Integer> listOfAttributesWhichShouldAlwaysBeThere = new ArrayList<Integer>();

    /**
     * changes 2 percent of the bits in the bitset
     * @param bitSet     the bitset to change
     * @return the slightly mutated bitset
     */
    protected BitSet changeBits(BitSet bitSet, int setPercentage) {
        Random r = new Random();
        boolean includesMoreThanXPercent = false;
        while (!includesMoreThanXPercent) {
            for (int i = 11; i < m_numAttribs - 2; i++) {              //starting from 11 because overlay fields are already set
                int chance = r.nextInt(100);                        //gives random Int between 0 (inclusive) and n (exclusive)
                if (chance < setPercentage)                                                       //set here the percent you want, just with the < symbol.
                    if (bitSet.get(i))
                        bitSet.set(i, false);
                    else
                        bitSet.set(i, true);
                if (includesMoreThanXPercentOfFeatures(bitSet, false, 1))         //making sure we dont drop below the desired % of features included after the mutation
                    includesMoreThanXPercent = true;
            }
        }
        return bitSet;
    }

    /**
     *
     *@param setPercentage Percentage of bits randomly being set to 1 for a start
     * @return BitSet with setPercentage of Bits set to 1
     */
    protected BitSet getStartSet(int setPercentage) {
        BitSet bitSet = new BitSet(m_numAttribs);
        Random r = new Random();
        boolean atLeastOneLagSet = false;
        boolean includesMoreThan25Percent = false;
        for (int i = 0; i < listOfAttributesWhichShouldAlwaysBeThere.size(); i++)
            bitSet.set(listOfAttributesWhichShouldAlwaysBeThere.get(i));
        for (int i = 11; i < m_numAttribs - 2; i++) {
            int chance = r.nextInt(100);
            if (chance < setPercentage)
                bitSet.set(i);
        }
        return bitSet;
    }
    /**
     * Returns a list of attributes (and or attribute ranges) as a String
     *
     * @return a list of attributes (and or attribute ranges)
     */
    protected String printGroup(BitSet tt) {
        for (int i = 0; i < m_numAttribs; i++)
            if (tt.get(i) == true)
                System.out.print((i + 1) + " ");
        System.out.println();
        return "";
    }

    public int getM_numAttribs() {
        return m_numAttribs;
    }

    public void setM_numAttribs(int m_numAttribs) {
        this.m_numAttribs = m_numAttribs;
    }

    public float howMuchPercentOfBitsAreDifferent(BitSet bitSet1, BitSet bitSet2) {
        float percent = 0;
        int differencesCounter = 0;
        for (int i = 0; i < m_numAttribs; i++)
            if ((bitSet1.get(i) & !bitSet2.get(i)) || (!bitSet1.get(i)) && bitSet2.get(i))
                differencesCounter++;
        return (float) differencesCounter * 100 / m_numAttribs;
    }

    public boolean includesMoreThanXPercentOfFeatures(BitSet bitSet, boolean print, int percentage) {
            int trueBits = 0;
        try {
            for (int i = 11; i < m_numAttribs - 2; i++)              //this interval makes sure we just consider the lags
                if (bitSet.get(i))
                    trueBits++;
            if (print)
                System.out.println("Including:" + ((float) trueBits / m_numAttribs) * 100 + "% of features.");
        }catch (Exception e){
            e.printStackTrace();
        }
        if ((float) trueBits / m_numAttribs > (float) percentage/100)
            return true;
        return false;
    }

    public SubsetHandler() { //TODO change this back after manual set
       /* listOfAttributesWhichShouldAlwaysBeThere.add(0);                    //time stamp and best overlay fields according to own experiments;
        listOfAttributesWhichShouldAlwaysBeThere.add(1);
        listOfAttributesWhichShouldAlwaysBeThere.add(2);
        listOfAttributesWhichShouldAlwaysBeThere.add(3);
        listOfAttributesWhichShouldAlwaysBeThere.add(6);
        listOfAttributesWhichShouldAlwaysBeThere.add(8);
        listOfAttributesWhichShouldAlwaysBeThere.add(9);*/

       listOfAttributesWhichShouldAlwaysBeThere.add(0);
       listOfAttributesWhichShouldAlwaysBeThere.add(1);
       listOfAttributesWhichShouldAlwaysBeThere.add(2);
       listOfAttributesWhichShouldAlwaysBeThere.add(5);                          //manual feature sets
        listOfAttributesWhichShouldAlwaysBeThere.add(7);
    }

}
