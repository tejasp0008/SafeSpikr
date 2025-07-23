`timescale 1ns/1ps
module tb_neuron;
reg clk, rst;
reg signed [15:0] membrane_potential;
wire spike;

neuron uut (
    .clk(clk),
    .rst(rst),
    .membrane_potential(membrane_potential),
    .spike(spike)
);

always #5 clk = ~clk;

initial begin
    $dumpfile("tb_neuron.vcd");
    $dumpvars(0, tb_neuron);

    clk = 0;
    rst = 1; membrane_potential = 0; #10;
    rst = 0; membrane_potential = 50; #10;
    membrane_potential = 100; #10;
    membrane_potential = 110; #10;
    membrane_potential = 80; #10;

    $finish;
end
endmodule

