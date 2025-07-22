module top_module (
    input clk,
    input rst,
    input spike_in,
    input signed [7:0] weight,
    input signed [7:0] grad_in,
    output spike,
    output signed [7:0] weight_updated
);

wire signed [15:0] acc_out;
wire spike_out;

mac mac_unit (
    .clk(clk),
    .rst(rst),
    .spike_in(spike_in),
    .weight(weight),
    .acc_in(16'd0),
    .acc_out(acc_out)
);

neuron neuron_unit (
    .clk(clk),
    .rst(rst),
    .membrane_potential(acc_out),
    .spike(spike_out)
);

optimizer optimizer_unit (
    .clk(clk),
    .rst(rst),
    .spike(spike_out),
    .weight_in(weight),
    .grad_in(grad_in),
    .weight_out(weight_updated)
);

assign spike = spike_out;

endmodule

