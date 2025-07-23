`timescale 1ns/1ps
module tb_top_module;

reg clk, rst;
reg spike_in;
reg signed [7:0] weight, grad_in;
wire spike;
wire signed [7:0] weight_updated;

// Instantiate the module
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

// Memory arrays
reg signed [7:0] weight_mem [0:0];     // Only 1 weight used
reg signed [7:0] grad_mem [0:0];       // Only 1 gradient (can be dummy)

// Simulation logic
initial begin
    $dumpfile("tb_top_module.vcd");
    $dumpvars(0, tb_top_module);

    // Load weight and gradient
    $readmemh("../verilog/weights_classifier_fc.txt", weight_mem);  // or adjust path
    grad_mem[0] = 8'h00;  // You can replace this with $readmemh if using file

    // Initialize
    clk = 0;
    rst = 1;
    spike_in = 0;
    weight = weight_mem[0];
    grad_in = grad_mem[0];
    #10;

    // Remove reset
    rst = 0;

    // Apply spikes
    spike_in = 1; #10;
    spike_in = 0; #10;
    spike_in = 1; #10;

    $finish;
end

endmodule
