// Rahmaan Lodhia
// Part 2: Branch Prediction Project
// Perceptron Implementation

#include <stdio.h>
#include <cassert>
#include <string.h>
#include <inttypes.h>
#include <cmath>

using namespace std;
#include "cbp3_def.h"
#include "cbp3_framework.h"

// this file includes two sample predictors
// one is a 64 KB gshare conditional predictor
// the other is a 64 KB indirect predictor indexed by (pc ^ history)

// rewind_marked is also used to show how to rewind the reader for multiple runs
// the predictor will use gshare in the first run
// the indirect predictor will be used in the second run

// NOTE: rewind_marked is only provided to help tuning work. the final
// submitted code should only include one run.


#define GSHARE_SIZE 18 // 256K 2-bit counters = 64 KB cost

#define GHISTORY_LENGTH 128// Length of GHR

// Definitions for perceptron predictor
#define PERC_SIZE 9    // Perceptron entry table with 6-bit weight counters
#define PERC_LENGTH 62 // Length of perceptron table entries 
#define PERC_THRESHOLD 1.93*PERC_LENGTH+14 // Perceptron training threshold

// Definitions for SWP predictor
#define SWP_SIZE 7 // Number of entries for SWP predictor, each entry is 6-bits
#define SWPL_1 20  // Length of SWP weights one entries
#define SWPL_2 16  // Length of SWP weights two entries
#define SWPL_3 29  // Length of SWP weights three entries
#define SWP_THRESHOLD 107 // SWP training threshold

// Definitions for Indirect predictor
#define IND_SIZE 13    // 16K 32-bit targets  = 64 KB cost

// predictor tables
int8_t   *gtable;
uint32_t *indtable;

// Peceptron table
int8_t  percTable[1 << PERC_SIZE][PERC_LENGTH];

// SWP tables
int8_t  wTaken[1 << (SWP_SIZE)][SWPL_1];
int8_t  wNotTaken[1 << (SWP_SIZE)][SWPL_1];
int8_t  wtOne[1 << (SWP_SIZE)][SWPL_2];
int8_t  wtTwo[1 << (SWP_SIZE - 1)][SWPL_3];

// Counters to score predictor performance for deciding conflicts
int8_t  pred1Count; // Count for perceptron predictor
int8_t  pred2Count; // Count for SWP predictor

// two branch history registers:
// the framework provids real branch results at fetch stage to simplify branch history
// update for predicting later branches. however, they are not available until execution stage
// in a real machine. therefore, you can only use them to update predictors at or after the
// branch is executed.
// in this sample code, we update predictors at retire stage where uops are processed
// in order to enable easy regneration of branch history.

// cost: depending on predictor size

// Branch History Registers
uint32_t brh_fetch;
uint32_t brh_retire;

// Global History Register
bool    *GHR;

// Global address register
uint32_t *GA;

// Speculative GHR
bool    *SGHR1; // SGHR for perceptron predictor
bool    *SGHR2; // SGHR for SWP predictor

// Speculative training register: keeps track of when a predictor needs to train
bool    *STrain1; // STrain for perceptron predictor
bool    *STrain2; // STrain for SWP predictor

// Predictor outputs
int32_t out1; // Output of perceptron
int32_t out2; // Output of SWP

// Table indicies
uint32_t ind1; // Index for perceptron tables
uint32_t ind2; // Index for SWP tables

// Counter for the number of branches that need to be retired
int8_t  speculCount;

// count number of runs
uint32_t runs;


// Function for hashing path and address of current branch to index into appropriate weight table
uint32_t getIndex(uint32_t pc, uint32_t path, uint32_t tableSize)
{
    pc = (pc) ^ (pc / (1 << tableSize));
    path = (path) ^ (path / (1 << tableSize));
    uint32_t index = (pc) ^ (path);
    index = index % (1 << tableSize);
    return index;
}


void PredictorInit() 
{
    runs = 2;
    gtable = new int8_t[1 << GSHARE_SIZE];
    indtable = new uint32_t[1 << IND_SIZE];
    GHR = new bool[GHISTORY_LENGTH];
    GA = new uint32_t[GHISTORY_LENGTH];
    SGHR1 = new bool[GHISTORY_LENGTH];
    SGHR2 = new bool[GHISTORY_LENGTH];
    STrain1 = new bool[GHISTORY_LENGTH];
    STrain2 = new bool[GHISTORY_LENGTH];

    assert(gtable && indtable);
    assert(GHR);
    assert(GA);
    assert(SGHR1);
    assert(SGHR2);
    assert(STrain1);
    assert(STrain2);
}

void PredictorReset() 
{
    // this function is called before EVERY run
    // it is used to reset predictors and change configurations

    if (runs == 0)
        printf("Predictor:gshare\nconfig: %i counters, %i KB cost\n", 1 << GSHARE_SIZE, (1 << GSHARE_SIZE) * 2 / 8 / 1024);
    else if (runs == 1)
        printf("Predictor:ind\nconfig: %i targets,  %i KB cost\n", 1 << IND_SIZE, (1 << IND_SIZE) * 4 / 1024);
    else if (runs == 2)
    {
        printf("Condtional Predictor: Hybrid Perceptron & SWP Predictor\n");
        printf("Hardware Costs:\n");
        printf("Perceptron Cost: %i entries, %i-bit global history, %i-bit weight counters\n", 1 << PERC_SIZE, PERC_LENGTH, 6); 
        printf("Totaling %i KB cost\n", (1 << PERC_SIZE) * PERC_LENGTH * 6 / 8 / 1024 );
        printf("SWP Cost: %i entry SWP, %i single entry table, %i single entry table, %i-bit weight counters\n", (1 << SWP_SIZE), (1 << SWP_SIZE), (1 << (SWP_SIZE-1)), 7);
        printf("Totaling %i KB cost\n", ((1 << SWP_SIZE) * 7 * 2 * SWPL_1 + (1 << SWP_SIZE) * 7 * SWPL_2 + (1 << (SWP_SIZE-1)) * 6 * SWPL_3) / 8 / 1024);
        printf("Indirect Predictor: TAP Predictor\n");
        printf("Hardware Costs: %i entries, %i KB cost\n", (1 << IND_SIZE), (1 << IND_SIZE) * 4/ 1024);
    }

    for (int i = 0; i < (1 << GSHARE_SIZE); i ++)
        gtable[i] = 0;

    for (int i = 0; i < (1 << IND_SIZE); i ++)
        indtable[i] = 0;

    for (int i = 0; i < (1 << PERC_SIZE); i++)
    {
        for (int j = 0; j < PERC_LENGTH; j++)
            percTable[i][j] = 0;
    }

    for (int i = 0; i < (1 << SWP_SIZE); i++)
    {
        for (int j = 0; j < SWPL_1; j++)
        {
            wTaken[i][j] = 0;
            wNotTaken[i][j] = 0;
        }

        for (int j = 0; j < SWPL_2; j++)
        {
            wtOne[i][j] = 0;
        }
    }

    for (int i = 0; i < (1 << (SWP_SIZE - 1)); i++)
    {
        for (int j = 0; j < SWPL_3; j++)
        {
            wtTwo[i][j] = 0;
        }
    }


    brh_fetch = 0;
    brh_retire = 0;

    speculCount = 0;
    pred1Count = 0;
    pred2Count = 0;
}

void PredictorRunACycle() 
{
    // get info about what uops are processed at each pipeline stage
    const cbp3_cycle_activity_t *cycle_info = get_cycle_info();

    // make prediction at fetch stage
    for (int i = 0; i < cycle_info->num_fetch; i++) 
    {
        uint32_t fe_ptr = cycle_info->fetch_q[i];
        const cbp3_uop_dynamic_t *uop = &fetch_entry(fe_ptr)->uop;

        if (runs == 0 && uop->type & IS_BR_CONDITIONAL) 
        {
            // get prediction
            uint32_t gidx = (brh_fetch ^ uop->pc) & ((1 << GSHARE_SIZE) - 1);
            bool gpred = (gtable[gidx] >= 0);

            // report prediction:
            // you need to provide direction predictions for conditional branches,
            // targets of conditional branches are available at fetch stage.
            // for indirect branches, you need to provide target predictions.
            // you can report multiple predictions for the same branch
            // the framework will use the last reported prediction to calculate 
            // misprediction penalty
            assert(report_pred(fe_ptr, false, gpred));
        }
        else if (runs == 1 && uop->type & IS_BR_INDIRECT) 
        {
            uint32_t gidx = (brh_fetch ^ uop->pc) & ((1 << IND_SIZE) - 1);
            uint32_t gpred = indtable[gidx];

            assert(report_pred(fe_ptr, false, gpred));
        }
        else if (runs == 2 && (uop->type & IS_BR_CONDITIONAL || uop->type & IS_BR_INDIRECT))
        {
            if (uop->type & IS_BR_CONDITIONAL)
            {
                // Get index of first entry of perceptron table
                ind1 = getIndex(uop->pc, GA[0], PERC_SIZE);

                // Calculate output of perceptron
                // Simulate bias bit
                out1 = percTable[ind1][0];
                for (int j = 1; j < PERC_LENGTH; j++)
                {
                    ind1 = getIndex(uop->pc, GA[j], PERC_SIZE);
                    if (GHR[j-1] == 1)
                        out1 += percTable[ind1][j];
                    else
                        out1 -= percTable[ind1][j];
                }

                // Make prediction for perceptron predictor
                bool gpred1 = (out1 >= 0);

                // Calculate output of SWP
                out2 = 0;
                // Calculate first 20 branches using seperated tables
                for (int j = 0; j < SWPL_1; j++)
                {
                    ind2 = getIndex(uop->pc, GA[j], SWP_SIZE);
                    if (GHR[j] == 1)
                        out2 += wTaken[ind2][j];
                    else
                        out2 += wNotTaken[ind2][j];
                }

                // Calculate next 16 branches using single table
                for (int j = 0; j < SWPL_2; j++)
                {
                    ind2 = getIndex(uop->pc, GA[SWPL_1 + j], SWP_SIZE);
                    if (GHR[SWPL_1 + j] == 1)
                        out2 += wtOne[ind2][j];
                    else
                        out2 -= wtOne[ind2][j];
                }

                // Calculate next 29 branches using smaller single table
                for (int j = 0; j < SWPL_3; j++)
                {
                    ind2 = getIndex(uop->pc, GA[SWPL_1 + SWPL_2 + j], SWP_SIZE-1);
                    if (GHR[SWPL_1 + SWPL_2 + j] == 1)
                        out2 += wtTwo[ind2][j];
                    else
                        out2 -= wtTwo[ind2][j];
                }

                bool gpred2 = (out2 >= 0);

                // Choose best predictor with a bias towards the perceptron predictor
                if (gpred1 != gpred2)
                {
                    pred1Count = 0;
                    pred2Count = 0;

                    // Count up the total number of correct predictions for each predictor for the last 10 branches
                    for (int j = 0; j < 10; j++)
                    {
                        pred1Count += (SGHR1[j] == GHR[j]);
                        pred2Count += (SGHR2[j] == GHR[j]);
                    }

                    // Pick prediction with a better score
                    if (pred1Count >= pred2Count)
                        assert(report_pred(fe_ptr, false, gpred1));
                    else
                        assert(report_pred(fe_ptr, false, gpred2));
                }
                else
                    assert(report_pred(fe_ptr, false, gpred1));

                // Shift all history registers
                for (int j = GHISTORY_LENGTH - 1; j > 0; j--)
                {
                    GHR[j] = GHR[j-1];
                    GA[j] = GA[j-1];
                    SGHR1[j] = SGHR1[j-1];
                    SGHR2[j] = SGHR2[j-1];
                    STrain1[j] = STrain1[j-1];
                    STrain2[j] = STrain2[j-1];
                }

                // Update registers
                GHR[0] = (uop->br_taken ? 1 : 0);
                GA[0] = uop->pc;
                SGHR1[0] = gpred1;
                SGHR2[0] = gpred2;

                //Check if training is needed
                if (out1 >= -PERC_THRESHOLD && out1 <= PERC_THRESHOLD)
                    STrain1[0] = 1;
                else
                    STrain1[0] = 0;

                if (out2 >= -SWP_THRESHOLD && out2 <= SWP_THRESHOLD)
                    STrain2[0] = 1;
                else
                    STrain2[0] = 0;

                // A branch has been fetched, so the speculation counter is increased
                speculCount++;
            }
            else if (uop->type & IS_BR_INDIRECT)
            {
                uint32_t gidx = (brh_fetch ^ uop->pc) & ((1 << IND_SIZE) - 1);
                uint32_t gpred = indtable[gidx];
                assert(report_pred(fe_ptr, false, gpred));
            }
        }

        

        // update fetch branch history
        if (uop->type & IS_BR_CONDITIONAL)
        {
            brh_fetch = (brh_fetch << 1) | (uop->br_taken ? 1 : 0);
        }
        else if (uop_is_branch(uop->type))
        {
            brh_fetch = (brh_fetch << 1) | 1;
        }

    }

    for (int i = 0; i < cycle_info->num_retire; i++) 
    {
        uint32_t rob_ptr = cycle_info->retire_q[i];
        const cbp3_uop_dynamic_t *uop = &rob_entry(rob_ptr)->uop;

        if (runs == 0 && uop->type & IS_BR_CONDITIONAL) 
        {
            uint32_t gidx = (brh_retire ^ uop->pc) & ((1 << GSHARE_SIZE) - 1);

            // update predictor
            bool t = uop->br_taken;
            if (t && gtable[gidx] < 1)
                gtable[gidx] ++;
            else if (!t && gtable[gidx] > -2)
                gtable[gidx] --;
        }
        else if (runs == 1 && uop->type & IS_BR_INDIRECT) 
        {
            uint32_t gidx = (brh_retire ^ uop->pc) & ((1 << IND_SIZE) - 1);
            indtable[gidx] = uop->br_target;
        }
        else if (runs == 2 && (uop->type & IS_BR_CONDITIONAL || uop->type & IS_BR_INDIRECT))
        {
            if (uop->type & IS_BR_CONDITIONAL)
            {
                // Get t
                bool t = uop->br_taken;

                // Train perceptron
                if ((t != SGHR1[speculCount-1]) || (STrain1[speculCount-1] == 1))
                {
                    //Check x0
                    ind1 = getIndex(uop->pc, GA[speculCount], PERC_SIZE);
                    //ind1 = (uop->pc) % (1 << PERC_SIZE);

                    if (t == true)
                    {
                        if (percTable[ind1][0] < 31)
                            percTable[ind1][0]++;
                    }
                    else
                    {
                        if (percTable[ind1][0] > -32)
                            percTable[ind1][0]--;
                    }

                    // Check rest of GHR
                    for(int j = 1; j < PERC_LENGTH; j++)
                    {
                        ind1 = getIndex(uop->pc, GA[speculCount+j], PERC_SIZE);
                        if (t == GHR[speculCount + j-1])
                        {
                            if (percTable[ind1][j] < 31)
                                percTable[ind1][j]++;
                        }
                        else
                        {
                            if (percTable[ind1][j] > -32)
                                percTable[ind1][j]--;
                        }
                    }
                }

                // Train SWP predictor
                if ((t != SGHR2[speculCount-1]) || (STrain2[speculCount-1] == 1))
                {
                    // Update seperated weight tables
                    for (int j = 0; j < SWPL_1; j++)
                    {
                        ind2 = getIndex(uop->pc, GA[j+speculCount], SWP_SIZE);
                        if (t == 1 && GHR[speculCount+j] == 1)
                        {
                            if (wTaken[ind2][j] < 63)
                                wTaken[ind2][j]++;
                        }
                        else if (t == 0 && GHR[speculCount+j] == 1)
                        {
                            if (wTaken[ind2][j] > -64)
                                wTaken[ind2][j]--;
                        }
                        else if (t == 1 && GHR[speculCount+j] == 0)
                        {
                            if (wNotTaken[ind2][j] < 63)
                                wNotTaken[ind2][j]++;
                        }
                        else if (t == 0 && GHR[speculCount+j] == 0)
                        {
                            if (wNotTaken[ind2][j] > -64)
                                wNotTaken[ind2][j]--;
                        }
                    }

                    // Update large weight table
                    for (int j = 0; j < SWPL_2; j++)
                    {
                        ind2 = getIndex(uop->pc, GA[j+speculCount+SWPL_1], SWP_SIZE);
                        if (t == GHR[speculCount+j+SWPL_1])
                        {
                            if (wtOne[ind2][j] < 63)
                                wtOne[ind2][j]++;
                        }
                        else
                        {
                            if (wtOne[ind2][j] > -64)
                                wtOne[ind2][j]--;
                        }
                    }

                    // Update smaller weight table
                    for (int j = 0; j < SWPL_3; j++)
                    {
                        ind2 = getIndex(uop->pc, GA[j+speculCount+SWPL_1+SWPL_2], SWP_SIZE-1);
                        if (t == GHR[speculCount+j+SWPL_1+SWPL_2])
                        {
                            if (wtTwo[ind2][j] < 63)
                                wtTwo[ind2][j]++;
                        }
                        else
                        {
                            if (wtTwo[ind2][j] > -64)
                                wtTwo[ind2][j]--;
                        }
                    }
                }

                // Decrement speculation count as branches are being retired
                speculCount--;
            }
            else if (uop->type & IS_BR_INDIRECT)
            {

                uint32_t gidx = (brh_retire ^ uop->pc) & ((1 << IND_SIZE) - 1);
                indtable[gidx] = uop->br_target;
            }

        }

        // update retire branch history
        if (uop->type & IS_BR_CONDITIONAL)
        {
            brh_retire = (brh_retire << 1) | (uop->br_taken ? 1 : 0);
        }
        else if (uop_is_branch(uop->type))
        {
            brh_retire = (brh_retire << 1) | 1;
        }
    }
}

void PredictorRunEnd() 
{
    runs ++;
    if (runs < 3) // set rewind_marked to indicate that we want more runs
        rewind_marked = true;
}

void PredictorExit() 
{
    delete [] gtable;
    delete [] indtable;
}
