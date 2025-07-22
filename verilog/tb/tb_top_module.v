`timescale 1ns/1ps
module tb_top_module;
reg clk, rst;
reg spike_in;
reg signed [7:0] weight, grad_in;
wire spike;
wire signed [7:0] weight_updated;

top_module uut (
    .clk(clk),
    .rst(rst),
    .spike_in(spike_in),
    .weight(weight),
    .grad_in(grad_in),
    .spike(spike),
    .weight_updated(weight_updated)
);

always #5 clk = ~clk;

initial begin
    $dumpfile("tb_top_module.vcd");
    $dumpvars(0, tb_top_module);

    clk = 0;
    rst = 1; spike_in = 0; weight = 8'd20; grad_in = 8'd3; #10;
    rst = 0;
    spike_in = 1; #10;
    spike_in = 0; #10;
    spike_in = 1; #10;

    $finish;
end
endmodule

