`timescale 1ns/1ps
module tb_optimizer;
reg clk, rst, spike;
reg signed [7:0] weight_in, grad_in;
wire signed [7:0] weight_out;

optimizer uut (
    .clk(clk),
    .rst(rst),
    .spike(spike),
    .weight_in(weight_in),
    .grad_in(grad_in),
    .weight_out(weight_out)
);

always #5 clk = ~clk;

initial begin
    $dumpfile("tb_optimizer.vcd");
    $dumpvars(0, tb_optimizer);

    clk = 0;
    rst = 1; spike = 0; weight_in = 10; grad_in = 2; #10;
    rst = 0;
    spike = 1; #10;
    spike = 0; #10;

    $finish;
end
endmodule
`
