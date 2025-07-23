`timescale 1ns/1ps
module tb_spike_loader;

  reg clk, rst;
  reg spike_in;
  reg signed [7:0] weight, grad_in;
  wire spike;
  wire signed [7:0] weight_updated;

  integer file, r;
  reg [127:0] line;
  reg [7:0] spike_val, weight_val, grad_val;

  // Instantiate top module
  top_module uut (
      .clk(clk),
      .rst(rst),
      .spike_in(spike_in),
      .weight(weight),
      .grad_in(grad_in),
      .spike(spike),
      .weight_updated(weight_updated)
  );

  // Clock generation
  always #5 clk = ~clk;

  initial begin
    $dumpfile("tb_spike_loader.vcd");
    $dumpvars(0, tb_spike_loader);

    clk = 0;
    rst = 1;
    #10 rst = 0;

    // Open spike data file
    file = $fopen("../outputs/spike_data.txt", "r");  // Adjust path if needed
    if (file == 0) begin
      $display("Error opening file.");
      $finish;
    end

    // Read input values
    while (!$feof(file)) begin
      r = $fscanf(file, "%d, %d, %d\n", spike_val, weight_val, grad_val);
      spike_in = spike_val;
      weight = weight_val;
      grad_in = grad_val;
      #10;
    end

    $fclose(file);
    $display("Simulation completed.");
    $finish;
  end

endmodule
