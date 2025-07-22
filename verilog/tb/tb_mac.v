`timescale 1ns/1ps

module tb_mac;
  // Inputs
  reg clk;
  reg rst;
  reg spike_in;
  reg signed [7:0] weight;
  reg signed [15:0] acc_in;

  // Output
  wire signed [15:0] acc_out;

  // Instantiate the MAC module
  mac uut (
    .clk(clk),
    .rst(rst),
    .spike_in(spike_in),
    .weight(weight),
    .acc_in(acc_in),
    .acc_out(acc_out)
  );

  // Clock generation
  always #5 clk = ~clk;  // 10ns period

  initial begin
    // Initialize signals
    $dumpfile("wave.vcd");   // for GTKWave
    $dumpvars(0, tb_mac);    // dump everything

    clk = 0;
    rst = 1;
    spike_in = 0;
    weight = 0;
    acc_in = 0;

    #10 rst = 0;

    // Test case 1: No spike, acc should hold
    acc_in = 16'sd10;
    weight = 8'sd3;
    spike_in = 0;
    #10;

    // Test case 2: Spike with positive weight
    spike_in = 1;
    #10;

    // Test case 3: Spike with negative weight
    acc_in = acc_out;    // continue accumulating
    weight = -8'sd2;
    #10;

    // Test case 4: No spike
    spike_in = 0;
    #10;

    // Done
    $finish;
  end

endmodule

